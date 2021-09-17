#!/usr/bin/python3
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
import time
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from function import *
from face_model import MobileFaceNet
from torchvision import transforms as trans
import json
from pyzbar.pyzbar import decode as decoder
import globalVar
import logging
import collections
from sklearn.decomposition import PCA
from sklearn.neighbors import RadiusNeighborsClassifier as KNC
import joblib

globalVar.initialize()

class FaceDetect():
    
    def __init__(self):
        torch.set_grad_enabled(False)
        self.datapath=os.getcwd()
        self.regFlag=False
        try:
            self.database=list(np.load(self.datapath+'/database.npy',allow_pickle=True))
            self.nodata=False
            self.inout=[False for i in range(len(self.database))]
            logging.info('data loaded')
        except Exception:
            logging.info('No Saved Data')
            self.nodata=True
        
        try:
            self.KNC = joblib.load('KNC.pkl' , mmap_mode ='r')
            self.PCA = joblib.load('PCA.pkl' , mmap_mode='r')
            logging.info("Classifier loaded")
        except Exception:
            self.KNC=KNC(radius=0.6,weights='distance')
            self.PCA = joblib.load('PCA.pkl' , mmap_mode ='r')
            logging.info('No Classifier')
        
        self.running=True
        self.reg=False
        self.facefeat=10
        self.recstkres=0
        self.count=0
        self.stime={}
        self.debounce={}
        self.recstk=[]
        self.embs=[]
        self.transform=trans.Compose([
                       trans.ToTensor(),
                       trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.loadnet()
        self.trigger()
                
    @torch.no_grad()    
    def loadnet(self):
        self.device = torch.device("cuda")
        self.recognizer= MobileFaceNet(512).to(self.device)
        self.recognizer.load_state_dict(torch.load('./weights/MobileFace_Net',map_location=lambda storage, loc: storage))
        self.recognizer.eval()
        logging.info('Recognizer loaded')
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)
        self.net = load_model(self.net, "./weights/FaceBoxesProd.pth",True)
        self.net.eval()
        self.net = self.net.to(self.device)
        logging.info('Detector loaded')
        cudnn.benchmark = True

    def trigger(self):
        timg=cv2.imread('triggered.jpg')
        a=self.getpoint(timg)
        for i in a:
            refsd= list(map(int,i))
            self.cropimg(refsd,timg)
            self.get_emb()

    def getpoint(self,frame):
        img = np.float32(frame)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        loc, conf = self.net(img)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        #ignore low scores
        inds = np.where(scores > 0.05)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, 0.3,force_cpu=False)
        dets = dets[keep, :]
        dets = dets[:750, :]
        return dets
    
    def stop(self):
        self.running=False
    
    def cropimg(self,point,img):
        self.crop_img = img[point[1]:point[3],point[0]:point[2]]
        if self.crop_img.shape[0]<=200 and self.crop_img.shape[1]<=180:
            return True
        try:
            self.crop_img=cv2.resize(self.crop_img,(112,112))
            return False
        except Exception:
            pass

    @torch.no_grad()
    def get_emb(self):
        if self.crop_img.shape==(112,112,3):
            self.crop_img=cv2.cvtColor(self.crop_img,cv2.COLOR_BGR2RGB)
            self.embs.append(self.recognizer(self.transform(self.crop_img).to(self.device).unsqueeze(0)).to('cpu').numpy().flatten())
        else:
            pass

    @torch.no_grad()
    def registration(self,registrationStart,alreadyRegistered,barcodeError):
        barcodescan=decoder(self.img_raw)
        if len(barcodescan)!=0:
            try:
                if self.nodata:
                    self.database=[]
                    self.inout=[]
                if barcodescan[0].type=='QRCODE':
                    data=json.loads(barcodescan[0].data.decode('utf-8'))
                    if data in self.database:
                        alreadyRegistered.emit()
                    else:
                        self.database.append(data)
                        self.inout.append(False)
                        self.count=0
                        self.reg=True
                        self.temps=[]
                        registrationStart.emit(data["name"])

            except ValueError :
                logging.info(barcodescan[0].data.decode('utf-8'))
                logging.info('Wrong barcode')
                barcodeError.emit()

    @torch.no_grad()
    def sync(self,datasync):
        try:
            for i in datasync:
                if i in self.database:
                    logging.info("sebelum : {}".format(self.database))
                    syncindex=self.database.index(i)
                    self.database.pop(syncindex)
                    self.inout=[False for i in range(len(self.database))]
                    startindex=syncindex*150
                    stopindex=startindex+150
                    Xdat=np.load('KNCparam.npy')
                    Ydat=np.load('KNCdat.npy')
                    logging.info("sebelum Xdat: {}".format(Xdat.shape))
                    logging.info("sebelum Ydat: {}".format(Ydat))
                    Xdat=np.delete(Xdat,[x for x in range(startindex,stopindex)],0)
                    logging.info(1)
                    Ydat=Ydat[:-150]
                    logging.info(2)
                    self.KNC.fit(Xdat,Ydat)
                    logging.info(3)
                    logging.info("sesudah : {}".format(self.database))
                    logging.info("sesudah Xdat: {}".format(Xdat.shape))
                    logging.info("sesudah Ydat: {}".format(Ydat))
                    np.save('database.npy',self.database)
                    np.save('KNCparam.npy',Xdat)
                    np.save('KNCdat.npy',Ydat)
                    joblib.dump(self.KNC,'KNC.pkl')
                else:
                    logging.info('Data Not Found')
        except Exception :
            return 'Sig_Sync_Error'


            

    @torch.no_grad()
    def recog(self,en,noData,unknownFace,result):
        if self.nodata:
            noData.emit()
        elif len(self.embs)==0:
            pass
        else:
            embs_PCA = self.PCA.transform(self.embs)
            try:
                pred=self.KNC.predict_proba(embs_PCA).flatten()
                neight=self.KNC.radius_neighbors(embs_PCA)
                xmax=neight[0][0][np.argmin(neight[0][0])]
                nearestneight=(neight[1][0][np.argmin(neight[0][0])])//150
                x=np.argmax(pred)
                probs=pred[x]*100
                posneigh=len(neight[0][0])*probs//100
                if (probs>=70 and posneigh>3 and x==nearestneight):
                    self.recstk.append(x)
                else:
                    self.recstk.append(-1)
            except ValueError:
                self.recstk.append(-1)
            if len(self.recstk)<5:
                pass
            else:
                c=collections.Counter(self.recstk)
                maxval = max(c, key=c.get)
                self.recstk=[]
                if maxval!=(-1) and c[maxval]>2:
                    if not(maxval in self.stime):
                        self.stime[maxval]=time.time()
                        self.debounce=True
                    else:
                        t=time.time()-self.stime[maxval]
                        if (t)<10:
                            self.debounce=False
                        else:
                            self.stime[maxval]=time.time()
                            self.debounce=True
                    if self.debounce:
                        self.inout[maxval]=not(self.inout[maxval])
                        result.emit((self.database[maxval],self.inout[maxval]))
                        #logging.info(time.time()-self.sptime)
                        self.sptime=0
                else:        
                    unknownFace.emit()
                    #logging.info(time.time()-self.sptime)
                    self.sptime=0
        self.recstkres=time.time()
            
    @torch.no_grad()
    def getfeature(self,regCount,registrationFinish):
        if self.count<150 and len(self.embs)!=0:
            logging.info('getting image feature')
            self.temps.append(self.embs[0])
            self.count+=1
            regCount.emit()
        if self.count==150:
            if self.nodata:
                Xdat=np.array(self.temps)
                Ydat=[len(self.database)-1]*150
                logging.info(Ydat)
                logging.info(len(Ydat))
                Xdat=self.PCA.transform(Xdat)
                logging.info(Xdat.shape)
                self.KNC.fit(Xdat,Ydat)
                np.save('KNCparam.npy',Xdat)
                np.save('KNCdat.npy',Ydat)
                joblib.dump(self.KNC,'KNC.pkl')
                self.nodata=False
            else:
                Xdat=np.load('KNCparam.npy',allow_pickle=True)
                self.temps=self.PCA.transform(np.array(self.temps))
                Xdat=np.append(Xdat,self.temps,axis=0)
                Ydat=np.load('KNCdat.npy',allow_pickle=True)
                Ydat=np.append(Ydat,np.array([len(self.database)-1]*150))
                self.KNC.fit(Xdat,Ydat)
                np.save('KNCparam.npy',Xdat)
                np.save('KNCdat.npy',Ydat)
                joblib.dump(self.KNC,'KNC.pkl')
            self.count=0
            self.reg=False
            np.save(self.datapath+'/database.npy',self.database) 
            registrationFinish.emit()

    def run(self,registrationStart,registrationFinish,alreadyRegistered,barcodeError,noData,unknownFace,result,regCount):
        self.sptime=0
        while self.running:
            self.img_raw=globalVar.image
            if self.sptime==0:
                self.sptime=time.time()
            if self.reg==False:
                self.registration(registrationStart,alreadyRegistered,barcodeError)
            dets=self.getpoint(self.img_raw)
            self.embs=[]
            for en,k in enumerate(dets):
                if k[4]<0.9:
                   continue
                text = "{:.4f}".format(k[4])
                b = list(map(int,k))
                checksize=self.cropimg(b,self.img_raw)
                if checksize:
                    continue
                self.get_emb()
                if self.reg and self.regFlag:
                    self.getfeature(regCount,registrationFinish)
                if  not self.reg:
                    self.recog(en,noData,unknownFace,result)
            restime=time.time()-self.recstkres
            if restime>=10:
                self.recstk=[]

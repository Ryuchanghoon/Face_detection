import cv2
import timeit

def videoDetector(cam,cascade):
    
    while True:
        
        start_t = timeit.default_timer()
        
        
    
        ret,img = cam.read()
        
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
      
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
         
        results = cascade.detectMultiScale(gray,            
                                           scaleFactor= 1.1,
                                           minNeighbors=5,  
                                           minSize=(20,20)  
                                           )
                                                                           
        for box in results:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
     
      
        terminate_t = timeit.default_timer()
        FPS = 'fps' + str(int(1./(terminate_t - start_t )))
        cv2.putText(img,FPS,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
        
        
           
        cv2.imshow('facenet',img)
        
        if cv2.waitKey(1) > 0: 
  
            break
 
def imgDetector(img,cascade):
    
    
    img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
   
    results = cascade.detectMultiScale(gray,            
                                       scaleFactor= 1.5,
                                       minNeighbors=5,  
                                       minSize=(20,20)  
                                       )        
        
    for box in results:
            
        x, y, w, h = box
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
    
     
    cv2.imshow('facenet',img)  
    cv2.waitKey(10000)

    



cascade_filename = 'haarcascade_frontalface_alt.xml'

cascade = cv2.CascadeClassifier(cascade_filename)

 
cam = cv2.VideoCapture('sample_final.mp4') # 비디오 파일

videoDetector(cam,cascade) # 영상 탐지

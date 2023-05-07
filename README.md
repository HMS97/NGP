## First do face recognition, then do whispering detection, speaker diarization, and combine video speech darilization.
## Then video captioning
## Then langchian find related frames interval


# 1. Face Recognition
```

from video import  VideoSeparator

vs = VideoSeparator(tolerance= [0.66, 0.66,0.68], max_encoding_length = 20,
                    path = "./Web/my-video.mp4")
# clip = vs.read_video()
vs.process_video_and_audio()
```

# 2. Video Captioning
```

python  blip2_caption.py
```


##  The above code is for data processing. once the data is processed, we can run the following code to run the web app

#  Set up the FASTAPI environment
```
uvicorn main:app --reload

```


# set up the HTTP server
```
cd Web/
http-server 

```

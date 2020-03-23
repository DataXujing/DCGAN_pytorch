import cv2
import os
 

# 图片路径
im_dir = './fake_g'
# 输出视频路径
video_dir = 'fake.mp4'
# 帧率(慢一点好看)
fps = 10
# 图片尺寸
img_size = (530, 530)
 
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

files = os.listdir(im_dir)
files.sort(key= lambda x:int(x[:-4]))

i = 0
for file in files:
    i += 1
    im_name = os.path.join(im_dir, file)
    print(file)
    frame = cv2.imread(im_name)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, "DCGAN(by XuJing) Epoch: 50,frame: {}".format(i), (20,40), font, 0.6, (255,0,255), 1)

    video_writer.write(frame)
    cv2.imshow('frame', frame)
    cv2.waitKey(20)
 
 
video_writer.release()
print('finish')

����flask��opencvʵʱ��Ƶ������������
opencvͨ��webcam���Ի�ȡ����ʵʱ��Ƶ�������������Ҫ����Ƶ������������������ã��Ϳ��Խ�����flask��ܹ���һ��ʵʱ��Ƶ����������Ȼ��������������ͨ�������������������������ȡ��̨�����ϵ�ʵʱ��Ƶ����**[��ƪ����]��https://blog.miguelgrinberg.com/post/video-streaming-with-flask��**�����ǳ���ϸ�����۽��ܺ;���ʵ�֣�������

������Ҫ˵����������flask�ṩ��Ƶ����ͨ��generator�������еģ����˽�Ŀ���ȥ�����ĵ�����Ͳ����彲�ˡ�flaskͨ����һ����������jpegͼƬ�����ʵ����Ƶ�������ַ�������motion JPEG���ô����ӳٺܵͣ����ǳ�������һ�㣬��Ϊjpegѹ��ͼƬ��������motion stream��̫���á�

multipart ģʽ
��Ҫ����һ������õ���ͼƬ���ǵ�ǰһ�δӶ��ﵽ������Ч������Ҫʹ����response��ʱ��ʹ��multipartģʽ��Multipart response�����¼�������ɣ�����multipart content���͵�header���ֽ���ŷָ��ĸ���part��ÿ��part�������ض���content���͡�multipart��Ƶ���Ľṹ���£�

HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg

<jpeg data here>
--frame
Content-Type: image/jpeg

<jpeg data here>
...
flask server
����ʵ�ִ��룺main.py

from flask import Flask, render_template, Response
import opencv

class VideoCamera(object):
    def __init__(self):
        # ͨ��opencv��ȡʵʱ��Ƶ��
        self.video = cv2.VideoCapture(0) 
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # ��Ϊopencv��ȡ��ͼƬ����jpeg��ʽ�����Ҫ��motion JPEGģʽ��Ҫ�Ƚ�ͼƬת���jpg��ʽͼƬ
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

app = Flask(__name__)

@app.route('/')  # ��ҳ
def index():
    # jinja2ģ�壬�����ʽ������index.html�ļ���
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        # ʹ��generator���������Ƶ���� ÿ�����������content������image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')  # �����ַ������Ƶ����Ӧ
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')   

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True�� port=5000)  
index.html

<html>
  <head>
    <title>Video Streaming Demonstration</title>
  </head>
  <body>
    <h1>Video Streaming Demonstration</h1>
    <img src="{{ url_for('video_feed') }}">
  </body>
</html>
ע��ͼƬ��ַ�ɴ������ڵ��ֵ������ָ��app�ĵڶ�����ַvideo_feed����multipartģʽ��������Ὣÿ������õ��ĵ�ַ�Դ����Ž��и��¡�

������
�����Ƶ��һֱ���ڵĻ������app�������Ƶ���ĵĿͻ��˵�������web worker��������ͬ����debugģʽ�£����������1��Ҳ����˵ֻ��һ����������ܹ�������Ƶ�������
���Ҫ�˷����־��޵Ļ���ʹ�û���Эͬ�������Ŀ�ܱ���gevent��������һ��worker�̷߳������ͻ��ˡ�

�ο�
https://blog.miguelgrinberg.com/post/video-streaming-with-flask
https://github.com/mattmakai/video-service-flask
http://www.chioka.in/python-live-video-streaming-example/
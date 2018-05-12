利用flask将opencv实时视频流输出到浏览器
opencv通过webcam可以获取本地实时视频流，但是如果需要将视频流共享给其他机器调用，就可以将利用flask框架构建一个实时视频流服务器，然后其他机器可以通过向这个服务器发送请求来获取这台机器上的实时视频流。**[这篇文章]（https://blog.miguelgrinberg.com/post/video-streaming-with-flask）**包含非常详细的理论介绍和具体实现，力荐！

首先需要说明的是这里flask提供视频流是通过generator函数进行的，不了解的可以去查下文档这里就不具体讲了。flask通过将一连串独立的jpeg图片输出来实现视频流，这种方法叫做motion JPEG，好处是延迟很低，但是成像质量一般，因为jpeg压缩图片的质量对motion stream不太够用。

multipart 模式
想要将后一次请求得到的图片覆盖到前一次从而达到动画的效果就需要使用在response的时候使用multipart模式。Multipart response由以下几部分组成：包含multipart content类型的header，分界符号分隔的各个part，每个part都具有特定的content类型。multipart视频流的结构如下：

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
具体实现代码：main.py

from flask import Flask, render_template, Response
import opencv

class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0) 
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

app = Flask(__name__)

@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')   

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True， port=5000)  
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
注：图片地址由大括号内的字典给出，指向app的第二个地址video_feed，在multipart模式下浏览器会将每次请求得到的地址对大括号进行更新。

局限性
如果视频流一直存在的话，这个app能输出视频流的的客户端的数量和web worker的数量相同，在debug模式下，这个数量是1，也就是说只有一个浏览器上能够看到视频流输出。
如果要克服这种局限的话，使用基于协同网络服务的框架比如gevent，可以用一个worker线程服务多个客户端。

参考
https://blog.miguelgrinberg.com/post/video-streaming-with-flask
https://github.com/mattmakai/video-service-flask
http://www.chioka.in/python-live-video-streaming-example/
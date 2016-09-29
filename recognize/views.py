from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from .forms import uploadVideoForm
from django.views.generic import View
from django.template import Context, Template
from .models import Document
import json
from .tasks import face_detection
import os
class uploadVideoView(View):
	template = 'index.html'

	def get(self, request):
		form = uploadVideoForm()
		return render(request, self.template, {"form":form})

	def post(self, request):
		form = uploadVideoForm(request.POST, request.FILES)
		file_name = None
		if form.is_valid():
			video = request.FILES.get('video')
			image = request.FILES.get('image')
			url  = 	request.POST.get('url')
			email = request.POST.get('email')
			if video:
				if image:
					newdoc = Document(video=video ,image=image)
					newdoc.save()
					face_detection.delay(email ,str(newdoc.video), str(newdoc.image))
				else :
					newdoc = Document(video=video)
					newdoc.save()
					face_detection.delay(email, video=str(newdoc.video))
			else :
				rtsp = False
				fileDir = os.path.dirname(os.path.realpath(__file__))
				if "http" in url:
					file_name = url.split('=')[-1]+'.mp4'
					path = fileDir + "/../videos/" + file_name
					os.system("youtube-dl -o "+ (path) + " " +url)
				else :
					rtsp = True
					file_name = "rtsp.mp4"
				if image:
					newdoc = Document(image=image)
					newdoc.save()
					face_detection.delay(email = email, video = 'videos/'+file_name, image = str(newdoc.image), url = url)
				else :
					face_detection.delay(email = email, video = 'videos/'+file_name, url = url)	
			
			# # Redirect to the document list after POST
			if request.META.get('HTTP_REFERER'):
				if video :
					file_name = str(newdoc.video).split('/')[-1].split('.')[0]
				else :
					file_name = file_name.split('.')[0]
			output = {'status' : 0,'count' : 0}
			fileDir = os.path.dirname(os.path.realpath(__file__))
			with open(fileDir + '/'+'json/'+file_name +'.json', 'w') as file:
					file.write(json.dumps(output))
		return render(request, self.template, {"form":form, "file_name" : file_name})

class checkStatusView(View):
	def post(self, request):
		fileDir = os.path.dirname(os.path.realpath(__file__))
		file_name = request.POST.get('file')
		# file_name = "obama"
		if file_name :
			try :
				with open(fileDir + '/'+'json/'+file_name +'.json','r') as file:
					response  = file.read()
					return HttpResponse(json.dumps(json.loads(response)),
						content_type = 'application/json'
					)
			except IOError :
				response = {
				'status' : '0','count' : '0'
				}
				return HttpResponse(json.dumps(response),
						content_type = 'application/json'
					)
		return HttpResponse(json.dumps({"status" : "404"}),
				content_type = 'application/json'
			)
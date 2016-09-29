from django.conf.urls import url
from . import views

urlpatterns = [
	url(r'^$', views.uploadVideoView.as_view()),
	url(r'^status/$', views.checkStatusView.as_view())
]
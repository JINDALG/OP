from django.core.mail import EmailMessage
from celery.decorators import task
from celery.utils.log import get_task_logger
from utils.demos.classifier_webcam import  main

logger = get_task_logger(__name__)

@task(name = "face_detection", ignore_result = True)
def face_detection(email, video, image = None, url = None):
	main(email=email, video_path=video, image_path=image, url=url)
	logger.info("process start")

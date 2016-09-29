from __future__ import unicode_literals

from django.db import models


class Document(models.Model):
	video = models.FileField(upload_to = 'videos/' , blank=True, null=True)
	image = models.FileField(upload_to = 'images/', blank=True, null=True)
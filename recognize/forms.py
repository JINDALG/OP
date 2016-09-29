from django import forms

class uploadVideoForm(forms.Form):
	image = forms.FileField(required=False)
	video = forms.FileField(required=False)
	url = forms.URLField(required=False)
	email = forms.EmailField(required=True)
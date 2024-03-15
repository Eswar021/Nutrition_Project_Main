from django import forms
from app1 import models

class nutrients_form(forms.ModelForm):
	class Meta:
		model = models.nutrients_model
		fields = "__all__"


class Recommendation_form(forms.ModelForm):
	class Meta:
		model = models.Recommendation_model
		fields = "__all__"


class Disease_form(forms.ModelForm):
	class Meta:
		model = models.Disease_model
		fields = "__all__"


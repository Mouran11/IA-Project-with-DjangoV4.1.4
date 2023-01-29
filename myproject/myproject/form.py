from django import forms

class ImageClassificationForm(forms.Form):
    image = forms.FileField()
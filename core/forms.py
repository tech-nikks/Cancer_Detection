from django import forms


class BreastCancerForm(forms.Form):

    radius = forms.FloatField(label='Mean Radius', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    texture = forms.FloatField(label='Mean Texture', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    perimeter = forms.FloatField(label='Mean Perimeter', min_value=0, max_value=200, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    area = forms.FloatField(label='Mean Area', min_value=0, max_value=2000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    smoothness = forms.FloatField(label='Mean Smoothness', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
from django.db import models
import pickle
import os
from Fruits_Vegs import settings
# Create your models here.
staticdir = settings.STATIC_DIR
print('--------')
print(staticdir)
print(os.listdir(staticdir))

with open(os.path.join(staticdir, 'PickleFiles', 'vitamins_info.pkl'), 'rb') as file:
    vitamins_info = pickle.load(file)

class nutrients_model(models.Model):
    Calcium = models.FloatField()
    Carbohydrate = models.FloatField()
    Cholesterol = models.FloatField()
    Fatty_acids_total_saturated = models.FloatField()
    Fatty_acids_total_trans = models.FloatField()
    Fiber = models.FloatField()
    Iron = models.FloatField()
    Protein  = models.FloatField()
    Sodium = models.FloatField()
    Sugars = models.FloatField()
    Fat = models.FloatField()


class Recommendation_model(models.Model):
    VITAMIN_CHOICES = [
        ('vitamin-a', 'Vitamin A'),
        ('vitamin-b1', 'Vitamin B1'),
        ('vitamin-b2', 'Vitamin B2'),
        ('vitamin-b3', 'Vitamin B3'),
        ('vitamin-b5', 'Vitamin B5'),
        ('vitamin-b6', 'Vitamin B6'),
        ('vitamin-b7', 'Vitamin B7'),
        ('vitamin-b9', 'Vitamin B9'),
        ('vitamin-b12', 'Vitamin B12'),
        ('vitamin-c', 'Vitamin C'),
        ('vitamin-d', 'Vitamin D'),
        ('vitamin-e', 'Vitamin E'),
        ('vitamin-k', 'Vitamin K'),
    ]

    Vitamin = models.CharField(max_length=20, choices=VITAMIN_CHOICES)


class Disease_model(models.Model):
    # Assuming 'Disease' is a field in your pickle file
    disease_choices = [(disease, disease) for disease in sorted(vitamins_info['Disease'])]
    # Adding a CharField for Disease
    disease = models.CharField(max_length=40, choices=disease_choices)
    def __str__(self):
        return self.disease
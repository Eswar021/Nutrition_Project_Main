# Generated by Django 5.0.1 on 2024-02-24 16:32

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='nutrients_model',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Calcium', models.FloatField()),
                ('Carbohydrate', models.FloatField()),
                ('Cholesterol', models.FloatField()),
                ('Fatty_acids_total_saturated', models.FloatField()),
                ('Fatty_acids_total_trans', models.FloatField()),
                ('Fiber_total_dietary', models.FloatField()),
                ('Iron', models.FloatField()),
                ('Protein', models.FloatField()),
                ('Sodium', models.FloatField()),
                ('Sugars', models.FloatField()),
                ('Total_lipid_fat', models.FloatField()),
            ],
        ),
    ]

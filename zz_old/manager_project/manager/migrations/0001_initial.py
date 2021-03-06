# Generated by Django 2.1.7 on 2019-03-11 15:41

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Manager',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
        ),
        migrations.CreateModel(
            name='Person',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=128)),
                ('birthday', models.DateTimeField()),
                ('sex', models.IntegerField(editable=False)),
                ('address_from', models.IntegerField()),
                ('current_address', models.IntegerField()),
                ('email', models.EmailField(max_length=254)),
            ],
        ),
        migrations.CreateModel(
            name='Worker',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('joined_at', models.DateTimeField()),
                ('quited_at', models.DateTimeField(blank=True, null=True)),
                ('manager', models.ForeignKey(on_delete='on_delete', to='manager.Manager')),
                ('person', models.ForeignKey(on_delete='on_delete', to='manager.Person')),
            ],
        ),
    ]

from django.db import models

# Create your models here.

class Person(models.Model):
    #kind of sex
    MAN = 0
    WOMAN = 1

    #kind of address
    HOKKAIDO = 0
    TOHOKU = 5
    TOKYO = 10
    CHIBA = 11
    KANAGAWA = 12
    SAITAMA = 13
    TOCHIGI = 14
    IBARAGI = 15
    CHUBU = 20
    KANSAI = 25
    CHUGOKU = 30
    SHIKOKU = 35
    KYUSHU = 40
    OKINAWA = 45

    #name
    name = models.CharField(max_length=128)
    #birthday
    birthday = models.DateTimeField()
    #sex
    sex = models.IntegerField(editable=False)
    #place of birth
    address_from = models.IntegerField()
    #current address
    current_address = models.IntegerField()
    #email address
    email = models.EmailField()

class Manager(models.Model):
    #kind of department
    DEP_ACCOUNTING = 0
    DEP_SALES = 5
    DEP_PRODUCTION = 10
    DEP_DEVELOPMENT = 15
    DEP_HR = 20
    DEP_FIN = 25
    DEP_AFFAIRS = 30
    DEP_PLANNING = 35
    DEP_BUSINESS = 40
    DEP_DISTR = 45
    DEP_IS = 50

class Worker(models.Model):
    person = models.ForeignKey('Person', 'on_delete')
    #when he/she was belonged to here
    joined_at = models.DateTimeField()
    #when he/she quited
    quited_at = models.DateTimeField(null=True, blank=True)
    #his/her boss
    manager = models.ForeignKey('Manager', 'on_delete')

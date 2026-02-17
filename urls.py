"""Cyber_Attack_Prediction URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from Cyber_Attack_Prediction_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index),
    path('index/',views.index),
    path('admin_login/',views.admin_login),
    path('admin_login_action/',views.admin_login_action),
    path('admin_home/',views.admin_home),
    path('logout/',views.logout),
    path('upload_dataset/',views.upload_dataset),
    path('upload_dataset_action/',views.upload_dataset_action),
    path('preprocess/',views.preprocess),
    path('build_model/',views.build_model),
    path('user_registration/',views.user_registration),
    path('user_registration_action/',views.user_registration_action),
    path('user_login/',views.user_login),
    path('user_login_action/',views.user_login_action),
    path('user_home/',views.user_home),
    path('enter_test_data/',views.enter_test_data),
    path('analysis_graphs/',views.analysis_graphs),

]

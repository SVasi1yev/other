from django.urls import path

from .views import handle_all_forum, handle_one_message


urlpatterns = [
    path('api/', handle_all_forum),
    path('api/<int:id_>/', handle_one_message)
]
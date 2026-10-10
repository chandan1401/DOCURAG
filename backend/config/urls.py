from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from django.views.generic import RedirectView
from backend.config.forms import EmailLoginForm
from backend.config.views import landing, signup, library, ask, history

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    path('', landing, name='landing'),
    path('library/', library, name='library'),
    path('ask/', ask, name='ask'),
    path('history/', history, name='history'),
    path('favicon.ico', RedirectView.as_view(url='/static/favicon.ico', permanent=True)),
    path('signup/', signup, name='signup'),
    path('login/', auth_views.LoginView.as_view(
        template_name='pages/login.html',
        authentication_form=EmailLoginForm,
        redirect_authenticated_user=True,
    ), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
]

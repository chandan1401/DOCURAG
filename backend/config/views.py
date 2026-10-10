from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from backend.config.forms import SignupForm


def landing(request):
    if request.user.is_authenticated:
        return redirect('ask')
    return render(request, 'pages/landing.html')


def signup(request):
    if request.user.is_authenticated:
        return redirect('ask')
    form = SignupForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        login(request, form.save())
        return redirect('ask')
    return render(request, 'pages/signup.html', {'form': form})


@login_required
def library(request):
    return render(request, 'pages/library.html')


@login_required
def ask(request):
    return render(request, 'pages/ask.html')


@login_required
def history(request):
    return render(request, 'pages/history.html')
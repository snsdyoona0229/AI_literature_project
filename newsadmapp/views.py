from django.shortcuts import render, redirect
from newsadmapp import models
from django.contrib.auth import authenticate
from django.contrib import auth
from django.contrib import messages
from django.contrib.auth.models import User
from django.views.decorators.csrf import ensure_csrf_cookie
from django import template
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage
from django.conf import settings
import math
import os
import sys
import shutil
from subprocess import call
import webbrowser
sys.path.append('/literature_project/AI_PART')
import geration_poet
import peot_Classification_model
import image_to_peot





page1 = 1

def index(request, pageindex=None):  #首頁
	global page1
	pagesize = 8
	newsall = models.NewsUnit.objects.all().order_by('-id')
	datasize = len(newsall)
	totpage = math.ceil(datasize / pagesize)
	if pageindex==None:
		page1 = 1
		newsunits = models.NewsUnit.objects.filter(enabled=True).order_by('-id')[:pagesize]
	elif pageindex=='1':
		start = (page1-2)*pagesize
		if start >= 0:
			newsunits = models.NewsUnit.objects.filter(enabled=True).order_by('-id')[start:(start+pagesize)]
			page1 -= 1
	elif pageindex=='2':
		start = page1*pagesize
		if start < datasize:
			newsunits = models.NewsUnit.objects.filter(enabled=True).order_by('-id')[start:(start+pagesize)]
			page1 += 1
	elif pageindex=='3':
		start = (page1-1)*pagesize
		newsunits = models.NewsUnit.objects.filter(enabled=True).order_by('-id')[start:(start+pagesize)]
	currentpage = page1
	return render(request, "index.html", locals())

def detail(request, detailid=None):  #詳細頁面
	unit = models.NewsUnit.objects.get(id=detailid)
	category = unit.catego
	title = unit.title
	pubtime = unit.pubtime
	nickname = unit.nickname
	message = unit.message
	unit.press += 1
	unit.save()
	return render(request, "detail.html", locals())

def login(request):  #登入
	messages = ''  #初始時清除訊息
	if request.method == 'POST':  #如果是以POST方式才處理
		name = request.POST['username'].strip()  #取得輸入帳號
		password = request.POST['password']  #取得輸入密碼
		user1 = authenticate(username=name, password=password)  #驗證
		if user1 is not None:  #驗證通過
			if user1.is_active:  #帳號有效
				auth.login(request, user1)  #登入
				return redirect('/adminmain/')  #開啟管理頁面
			else:  #帳號無效
				messages = '帳號尚未啟用！'
		else:  #驗證未通過
			messages = '登入失敗！'
	return render(request, "login.html", locals())

def logout(request):  #登出
	auth.logout(request)
	return redirect('/index/')

def adminmain(request, pageindex=None):  #管理頁面
	global page1
	pagesize = 8
	newsall = models.NewsUnit.objects.all().order_by('-id')
	datasize = len(newsall)
	totpage = math.ceil(datasize / pagesize)
	if pageindex==None:
		page1 = 1
		newsunits = models.NewsUnit.objects.order_by('-id')[:pagesize]
	elif pageindex=='1':
		start = (page1-2)*pagesize
		if start >= 0:
			newsunits = models.NewsUnit.objects.order_by('-id')[start:(start+pagesize)]
			page1 -= 1
	elif pageindex=='2':
		start = page1*pagesize
		if start < datasize:
			newsunits = models.NewsUnit.objects.order_by('-id')[start:(start+pagesize)]
			page1 += 1
	elif pageindex=='3':
		start = (page1-1)*pagesize
		newsunits = models.NewsUnit.objects.order_by('-id')[start:(start+pagesize)]
	currentpage = page1
	return render(request, "adminmain.html", locals())

def newsadd(request):  #新增資料
	message = ''  #清除訊息
	category = request.POST.get('news_type', '')  #取得輸入的類別
	subject = request.POST.get('news_subject', '')
	editor = request.POST.get('news_editor', '')
	content = request.POST.get('news_content', '')
	ok = request.POST.get('news_ok', '')
	context =''
	category_p = ''
	c = 0
	category_poet = ["AChiese", "BChiese", "CChiese", "DChiese","EChiese","FChiese","GChiese","HChiese"]
	poet_style = ["創世紀詩詞風格", "原住民詩詞風格", "客家詩詞風格", "新月詩詞風格","新詩詩詞風格","現代詩詞風格","笠詩詞風格","藍星詩詞風格"]
    
	if subject=='' or editor=='' or content=='':  #若有欄位未填就顯示訊息
		message = '每一個欄位都要填寫...'
	else:
		if ok=='v_01':  #根據ok值設定enabled欄位
			enabled = True
			context = geration_poet.peot_generation(str(category),str(content))
			c = category_poet.index(str(category))
			category_p = str(poet_style[c])
			category_class = str(poet_style[peot_Classification_model.predict(str(content))[2]])
			editor = "null"            
		if ok=='v_02':
			call(["python", "/literature_project/AI_PART/peot_Classification_model.py"])
			enabled = True
			peot_Classification_model.predict(str(content))
			context = str(content) +"[["+str(peot_Classification_model.predict(str(content))[0])+"]]風格百分比"
			category_p = str(poet_style[peot_Classification_model.predict(str(content))[2]])
			editor = "null"
		if ok=='v_03':
			c = category_poet.index(str(category))
			category_p = str(poet_style[c])
            
			call(["python", "/literature_project/AI_PART/image_to_peot.py"])
			enabled = True
			input_string = request.POST.get('url', '')
			input_string = str(input_string).replace("\\", "\\\\")
			context = image_to_peot.image_to_text_gpt(input_string,str(category))
			shutil.copy(input_string, "/literature_project/media") 
			#input_string = str(input_string).replace("\\\\","/")
			str_x = input_string.split("\\")
			editor = str_x[len(str_x)-1]
			

            
		unit = models.NewsUnit.objects.create(catego=category_p, nickname=editor, title=subject, message=context, enabled=enabled, press=0)
		unit.save()
		return redirect('/adminmain/')
	return render(request, "newsadd.html", locals())

def newsedit(request, newsid=None, edittype=None):  #修改資料
	unit = models.NewsUnit.objects.get(id=newsid)  #讀取指定資料
	categories = ["公告", "更新", "活動", "其他"]
	if edittype == None:  #進入修改頁面,顯示原有資料
		type = unit.catego
		subject = unit.title
		editor = unit.nickname
		content = unit.message
		ok = unit.enabled
	elif edittype == '1':  #修改完畢,存檔
		category = request.POST.get('news_type', '')
		subject = request.POST.get('news_subject', '')
		editor = request.POST.get('news_editor', '')
		content = request.POST.get('news_content', '')
		ok = request.POST.get('news_ok', '')
		if ok=='yes':
			enabled = True
		else:
			enabled = False
		unit.catego=category
		unit.nickname=editor
		unit.title=subject
		unit.message=content
		unit.enabled=enabled
		unit.save()
		return redirect('/adminmain/')
	return render(request, "newsedit.html", locals())

def newsdelete(request, newsid=None, deletetype=None):  #刪除資料
	unit = models.NewsUnit.objects.get(id=newsid)  #讀取指定資料
	if deletetype == None:  #進入刪除頁面,顯示原有資料
		type = str(unit.catego.strip())
		subject = unit.title
		editor = unit.nickname
		content = unit.message
		date = unit.pubtime
	elif deletetype == '1':  #按刪除鈕
		unit.delete()
		return redirect('/adminmain/')
	return render(request, "newsdelete.html", locals())


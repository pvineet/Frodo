
# coding: utf-8

import flickrapi
import urllib.request
import datetime
from os import mkdir

#API key and secret for flickr APIs
api_key = u'2b16cd87ab30970d4615de64cc7513d9'
api_secret = u'eed47a4955e774d8'
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
# List of classes
classes = ['airplane', 'car', 'bird', 'cat', 'flower', 'dog', 'person' , 'frog', 'ship', 'horse']
#Dictionary to choose image size
size = {'75x75':'s','150x150':'q'}
#base_path = '/Volumes/Seagate Slim Drive/Vineet/Flickr/'
#Base path to store the downloaded data, change is needed
base_path = '/Users/Vineet/Documents/Courses/ELEN 297/Flickr/'
# Add a time stamp to the output directory
timestamp = str(datetime.datetime.now()).replace(' ','-').split('.')[0]
path = base_path+timestamp+'/'
# Number of images for train test and validation set
sets = {'train':100, 'validation':20, 'test':20}
NUM_PHOTOS = (sum(sets.values()))

def get_photo_urls(tag,num):
    photos = flickr.photos.search(tags=tag,per_page='200')
    urls = []
    cnt = 0
    print(str(len(photos['photos']['photo']))+" photos of "+tag)
    #https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{o-secret}_o.(jpg|gif|png)
    for photo in (photos['photos']['photo']):
        farm_id = str(photo['farm'])
        server_id = str(photo['server'])
        id = str(photo['id'])
        secret = str(photo['secret'])
        urls.append('https://farm'+farm_id+'.staticflickr.com/'+server_id+'/'+id+'_'+secret+'_q.jpg')
        cnt += 1
        if cnt >= num:
            break
    return urls

def download_photos(urls, path, tag,):
    cnt = 0
    for url in urls:
        if cnt < sets['train']:
            urllib.request.urlretrieve(url, path+'/train/'+tag+'/'+str(cnt)+'.jpg')
        elif cnt < sets['validation']+sets['train']:
            urllib.request.urlretrieve(url, path+'/validation/'+tag+'/'+str(cnt)+'.jpg')
        else:
            urllib.request.urlretrieve(url, path+'/test/'+tag+'/'+str(cnt)+'.jpg')
        cnt+=1

def make_dirs(sets, classes):
    for dataset in sets.keys():
        mkdir(path+'/'+dataset+'/')
    for dataset in sets.keys():
        for classtype in classes:
            mkdir(path+'/'+dataset+'/'+classtype+'/')

# Make the path directory
mkdir(path)
# Make dirs for individual classes
make_dirs(sets, classes)

for classtype in classes:
    #Get photos of the particular class.
    urls = get_photo_urls(classtype, NUM_PHOTOS)
    download_photos(urls, path, classtype)
print("Done!")


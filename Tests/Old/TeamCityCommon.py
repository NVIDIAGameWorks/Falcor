import urllib.parse
import urllib.request
import getpass
import argparse
import ssl
import base64
import xml.etree.ElementTree as ET

server_url = 'https://teamcity.nvidia.com/'
project_url =  'app/rest/builds/?locator=project:Falcor,count:1000'

sslContext = ssl.SSLContext()
sslContext.verify_mode = ssl.CERT_REQUIRED
sslContext.check_hostname = True
sslContext.load_default_certs()

connectionTimeOut = 1000;

def get_request(subpage):
    r = urllib.request.Request(server_url + subpage, headers= authentification_header)
    r.timeout = connectionTimeOut
    handler = urllib.request.HTTPSHandler(context=sslContext)
    
    try:
        return handler.https_open(r)
    except urllib.error.HTTPError as httpErr:
        reason = httpErr.reason
        print('Error on urlopen ' + str(httpErr.code) + ' . ' + httpErr)

def post_request(subpage, post_data, content_type_str):
    
    local_authentification_headers = authentification_header
    if content_type_str:
        local_authentification_headers['Content-Type'] = content_type_str
    
    r = urllib.request.Request(server_url + subpage, data=post_data, headers= local_authentification_headers)
    r.timeout = connectionTimeOut
    handler = urllib.request.HTTPSHandler(context=sslContext)
    
    try:
        return handler.https_open(r)
    except urllib.error.HTTPError as httpErr:
        reason = httpErr.reason
        print('Error on urlopen ' + str(httpErr.code) + ' . ' + httpErr)

authentification_header = {}

def connect(username, password = ''):
    if not username:
        raise ConnectError('No username provided for teamcity connection')
    
    if not password:
        getPassPrompt = 'Enter teamcity password for user ' + username + ':'
        password = getpass.getpass(prompt=getPassPrompt )
    
    auth = base64.b64encode(':'.join([username, password]).encode()).decode('ascii')
    global authentification_header
    authentification_header = {'Authorization': ('Basic %s:' % auth)}
    
    r = urllib.request.Request(server_url, headers= authentification_header)
    r.timeout = connectionTimeOut
    handler = urllib.request.HTTPSHandler(context=sslContext)
    
    try:
        handler.https_open(r)
    except urllib.error.HTTPError as httpErr:
        reason = httpErr.reason
        print('Error on urlopen ' + str(httpErr.code) + ' . ' + httpErr)
        print('Failed to start_build')


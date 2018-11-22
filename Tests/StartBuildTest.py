import urllib.parse
import urllib.request
import urllib.error
import getpass
import argparse
import xml.etree.ElementTree as ET
from TeamCityCommon import connect
from TeamCityCommon import server_url 
from TeamCityCommon import project_url
import base64
import os
import GetBuildStatus

default_xml_file = './build.xml'
queue_url = 'app/rest/buildQueue?locator=project:Falcor'

def start_build_internal(username, password, xml_data):
    buildQueue_url = server_url + queue_url
    
    auth = username + ":" + password
    base64string = base64.encodestring(auth.encode() )
    r = urllib.request.Request(buildQueue_url, xml_data, {'Content-Type' : 'application/xml', 'Authorization' : ('Basic %s' % base64string) })
    
    try:
        returnData = urllib.request.urlopen(r)
    except urllib.error.HTTPError as httpErr:
        reason = httpErr.reason
        print('Error on urlopen ' + str(httpErr.code) + ' . ' + httpErr)
        print('Failed to start_build')
        
    return

def start_build(username, password, xml_file_path, branch_name, git_path, tests_directory, buildTypeId):
    file = open(xml_file_path, 'rt')
    data = file.read()
    file.close()
    
    # insert branch name into 
    xml = ET.fromstring(data)
    
    print('Starting remote build with id: ' + buildTypeId)
    
    # insert branch name into correct location
    for data in xml.iter():
        if data.get('branch'):
            data.set('branch', branch_name)
        if buildTypeId:
            if data.get('id'):
               data.set('id', buildTypeId)
        for param in data.findall('property'):
            if ( param.get('name') == 'branchname'):
                param.set('value', branch_name)
            if param.get('name') == 'tests_directory':
                if tests_directory:
                    param.set('value', '--tests_directory ' + tests_directory )
            if (param.get('name') == 'vcsRoot'):
                # get vsc root list from teamcity
                # find id from config
                please = GetBuildStatus.get_vcs_instances()
                string = str(please.read().decode())
                vcs_xml = ET.fromstring(string)
                vcs_id = ''
                
                for node in vcs_xml.iter():
                    for instance in node.findall('vcs-root-instance'):
                        if instance.get('name').startswith(git_path):
                            vcs_id = instance.get('vcs-root-id')
                            set = True
                            break;
                
                param.set('value', vcs_id)
    
    # convert back to string to be sent in post request
    xml_data = ET.tostring(xml)
    start_build_internal(username, password, xml_data)

def main():
    # Argument Parser.
    parser = argparse.ArgumentParser()
    
    # Adds argument for username to connect as
    parser.add_argument('-u', '--username', action='store', help='UserName To Connect.');
    
    # Adds argument for specifying wich xml to use for the build settings
    parser.add_argument('-xml', '--xml_filename', action='store', help='XML file to send in POST request for build.')
    
    # Parse the Arguments.
    args = parser.parse_args()
    
    if args.username:
        username = args.username
    else:
        username = input('Enter username for teamcity.nvidia.com: ')
    
    start_build(username, xml_file = default_xml_file)

if __name__ == '__main__':
    main()
import urllib.parse
import urllib.request
import getpass
import argparse
import xml.etree.ElementTree as ET
from TeamCityCommon import connect
from TeamCityCommon import server_url 
from TeamCityCommon import project_url
from time import sleep
import os
import sys
import webbrowser

buildLogUrlString = '&tab=buildLog'
artifactsUrlString = '&tab=artifacts'

class ConnectError(Exception):
    pass

def get_build_types():
    r = urllib.request.Request(server_url + 'app/rest/projects/id:Falcor/buildTypes')
    return urllib.request.urlopen(r)

def get_builds(suffix):
    r = urllib.request.Request(project_url + suffix)# urllib.parse.urlencode(get_fields).encode())
    return urllib.request.urlopen(r)

def get_all_builds():
    return get_builds(',running:any')

def get_running_builds():
    return get_builds(',state:running')

def get_queued_builds():
    return get_builds(',state:queued')

def get_vcs_instances():
    r = urllib.request.Request(server_url + 'app/rest/vcs-root-instances/?locator=project:Falcor')
    return urllib.request.urlopen(r)

def get_running_buildIds():
    buildIds = []
    data = get_running_builds();
    dataString = str(data.read().decode())
    xmldata = ET.fromstring(dataString);
    for node in xmldata.iter():
            for build in node.findall('build'):
                buildIds.append(build.get('id'))
    
    return  buildIds

def connect_and_get_build_status(username):
    connect(username)
    
    op = get_running_builds() 
    dataString = str(op.read().decode())
    
    #  data for all running tests
    xmldata = ET.fromstring(dataString);
    
    # go through each running config and get status for each test
    for node in xmldata.iter():
        for build in node.findall('build'):
            print('Build: ' + build.get('buildTypeId') + ' is running.' )

# assumes connect is already called
#once build is dispatched, wait for status to finish for each test
def wait_for_running_builds(runningBuilds):
    # get status
    period = 2.0
    
    running_builds = get_running_buildIds()
    
    if len(runningBuilds) == 0:
        print('No currently running builds on TeamCity!!')
        return False;
    
    print( 'Tests are currently running.' )
    print('\n Builds Sent:')
    print(runningBuilds)
    
    # check every period on status
    while True:
        #  data for all running tests
        op = get_all_builds()
        dataString = str(op.read().decode())
        xmldata = ET.fromstring(dataString);
        
        # go through each running config and get status for each test
        for node in xmldata.iter():
            for build in node.findall('build'):
                buildTypeId = build.get('buildTypeId')
                buildId = build.get('id')
                if (buildTypeId in runningBuilds) and (buildId in running_builds):
                    if build.get('state') != 'running':
                        runningBuilds.remove(buildTypeId)
                        running_builds.remove(buildId)
                        buildStatus = build.get('status')
                        print('Build ' + build.get('buildTypeId') + ' had stopped running with status ' + buildStatus + '. ')
                        if buildStatus == 'FAILURE':
                            buildLogUrl = build.get('webUrl') + buildLogUrlString
                            webbrowser.open( buildLogUrl )
                        else:
                            # open artifacts
                            buildLogUrl = build.get('webUrl') + artifactsUrlString
                            webbrowser.open( buildLogUrl )
        
        if len(running_builds) == 0:
            break
        
        sleep(period)
    
    print('All tests completed.')
    return True;


def main():
    
    # Argument Parser.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-u', '--username', action='store', help='UserName To Connect.');
    
    # Parse the Arguments.
    args = parser.parse_args()
    
    if args.username:
        username = args.username
    else:
        username = input('Enter username for teamcity.nvidia.com: ')
    
    connect_and_get_build_status(username)

if __name__ == '__main__':
    main()
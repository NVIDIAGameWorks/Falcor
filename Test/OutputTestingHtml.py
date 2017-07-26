import os
import TestingUtil as testingUtil

def lowLevelTestResultToHTML(result):
    html = '<tr>'
    if int(result.Crashed) > 0:
        html = '<tr bgcolor="yellow">\n'
        font = '<font>'
    elif int(result.Failed) > 0:
        html = '<tr bgcolor="red">\n'
        font = '<font color="white">'
    else:
        font = '<font>'
    html += '<td>' + font + result.Name + '</font></td>\n'
    html += '<td>' + font + str(result.Total) + '</font></td>\n'
    html += '<td>' + font + str(result.Passed) + '</font></td>\n'
    html += '<td>' + font + str(result.Failed) + '</font></td>\n'
    html += '<td>' + font + str(result.Crashed) + '</font></td>\n'
    html += '</tr>\n'
    return html

def getLowLevelTestResultsTable(slnInfo):
    if slnInfo.lowLevelResultList:
        html = '<table style="width:100%" border="1">\n'
        html += '<tr>\n'
        html += '<th colspan=\'5\'>Low Level Test Results</th>\n'
        html += '</tr>\n'
        html += '<tr>\n'
        html += '<th>Test</th>\n'
        html += '<th>Total</th>\n'
        html += '<th>Passed</th>\n'
        html += '<th>Failed</th>\n'
        html += '<th>Crashed</th>\n'
        for result in slnInfo.lowLevelResultList:
            html += lowLevelTestResultToHTML(result)
        html += '</table>\n'
        return html
    else:
        return ''

def systemTestResultToHTML(result):
    #if missing data for both load time and frame time, no reason for table entry
    if ((result.LoadTime == 0 or result.RefLoadTime == 0) and
        (result.AvgFrameTime == 0 or result.RefAvgFrameTime == 0)):
        return ''

    html = '<tr>'
    html += '<td>' + result.Name + '</td>\n'
    #if dont have real load time data, dont put it in table
    if result.LoadTime == 0 or result.RefLoadTime == 0:
        html += '<td></td><td></td><td></td>'
    else:
        html += '<td>' + str(result.LoadErrorMargin) + '</td>\n'
        if result.LoadTime > (result.RefLoadTime + result.LoadErrorMargin):
            html += '<td bgcolor="red"><font color="white">'
        elif result.LoadTime < result.RefLoadTime:
            html += '<td bgcolor="green"><font color="white">'
        else:
            html += '<td><font>'
        html += str(result.LoadTime) + '</font></td>\n'
        html += '<td>' + str(result.RefLoadTime) + '</td>\n'

    #if dont have real frame time data, dont put it in table
    if result.AvgFrameTime == 0 or result.RefAvgFrameTime == 0:
        html += '<td></td><td></td><td></td>'
    else:
        html += '<td>' + str(result.FrameErrorMargin * 100) + '</td>\n'
        compareResult = testingUtil.marginCompare(result.AvgFrameTime, result.RefAvgFrameTime, result.FrameErrorMargin)
        if compareResult == 1:
            html += '<td bgcolor="red"><font color="white">'
        elif compareResult == -1:
            html += '<td bgcolor="green"><font color="white">'
        else:
            html += '<td><font>'
        html += str(result.AvgFrameTime) + '</font></td>\n'
        html += '<td>' + str(result.RefAvgFrameTime) + '</td>\n'

    html += '</tr>\n'
    return html

def getSystemTestResultsTable(slnInfo):


    if slnInfo.systemResultList:

        entries = ''
        for result in slnInfo.systemResultList:
            entries += systemTestResultToHTML(result)

        html = ''
        if (entries != ''):
            html += '<table style="width:100%" border="1">\n'
            html += '<tr>\n'
            html += '<th colspan=\'7\'>System Test Results</th>\n'
            html += '</tr>\n'
            html += '<th>Test</th>\n'
            html += '<th>Load Time Error Margin Secs</th>\n'
            html += '<th>Load Time</th>\n'
            html += '<th>Ref Load Time</th>\n'
            html += '<th>Frame Time Error Margin %</th>\n'
            html += '<th>Avg Frame Time</th>\n'
            html += '<th>Ref Frame Time</th>\n'
            html += entries
            html += '</table>\n'

        return html

    else:
        return ''

def getImageCompareResultsTable(slnInfo):
    if slnInfo.systemResultList:
        maxCols = 0
        #table needs max num of screenshots plus one columns
        for result in slnInfo.systemResultList:
            if len(result.CompareImageResults) > maxCols:
                maxCols = len(result.CompareImageResults)

        if(maxCols > 0):
            html = '<table style="width:100%" border="1">\n'
            html += '<tr>\n'
            html += '<th colspan=\'' + str(maxCols + 1) + '\'>Image Compare Tests</th>\n'
            html += '</tr>\n'
            html += '<th>Test</th>\n'
            for i in range (0, maxCols):
                html += '<th>SS' + str(i) + '</th>\n'
            for result in slnInfo.systemResultList:
                if len(result.CompareImageResults) > 0:
                    html += '<tr>\n'
                    html += '<td>' + result.Name + '</td>\n'
                    for compare in result.CompareImageResults:
                        if float(compare) > testingUtil.gDefaultImageCompareMargin or float(compare) < 0:
                            html += '<td bgcolor="red"><font color="white">' + str(compare) + '</font></td>\n'
                        else:
                            html += '<td>' + str(compare) + '</td>\n'
                    html += '</tr>\n'
            html += '</table>\n'
            return html
        else:
            return ''
    else:
        return ''


def getMemoryCompareResultsTable(slnInfo):


    if slnInfo.systemResultList:

        maxCols = 7
        hasFrameCheck = False
        html = ''

        for result in slnInfo.systemResultList:
            if len(result.CompareMemoryFrameResults) > 0:
                hasFrameCheck = True

        if hasFrameCheck :
            html += '<table style="width:100%" border="1">\n'
            html += '<tr>\n'
            html += '<th colspan=\'' + str(maxCols) + '\'> Memory Frame Checks </th> \n'
            html += '</tr>\n'
            html += '<th> Test </th> \n' + '<th> Start Frame </th> \n' + '<th> End Frame </th> \n' + '<th> Percent Difference </th> \n' + '<th> Start Frame Memory </th> \n' + '<th> End Frame Memory </th> \n' + '<th> Difference </th> \n'

            for result in slnInfo.systemResultList:
                html += '<tr>\n'
                html += '<td>' + result.Name + '</td>\n'
                for compare in result.CompareMemoryFrameResults:
                    if(compare[2] > testingUtil.gMemoryPercentCompareMargin):
                        html += '<font color="white">' + '<td bgcolor="red">' + str(compare[0]) + '</td>' + '<td bgcolor="red">' + str(compare[1]) + '</td>' + '<td bgcolor="red">' + str(compare[2]) + '</td>' + '<td bgcolor="red">' + str(compare[3]) + '</td>' + '<td bgcolor="red">' + str(compare[4]) + '</td>' + '<td bgcolor="red">' + str(compare[5]) + '</td>' + '</font>'
                    else:
                        html += '<td>' + str(compare[0]) + '</td>' + '<td>' + str(compare[1]) + '</td>' + '<td>' + str(compare[2]) + '</td>' + '<td>' + str(compare[3]) + '</td>' + '<td>' + str(compare[4]) + '</td>' + '<td>' + str(compare[5]) + '</td>'

                html += '</tr>\n'

            html += '</table>\n'
            html += '<br><br>'

        hasTimeCheck = False

        for result in slnInfo.systemResultList:
            if len(result.CompareMemoryTimeResults) > 0:
                hasTimeCheck = True

        if hasTimeCheck :
            html += '<table style="width:100%" border="1">\n'
            html += '<tr>\n'
            html += '<th colspan=\'' + str(maxCols) + '\'> Memory Time Checks </th> \n'
            html += '</tr>\n'
            html += '<th> Test </th> \n' + '<th> Start Time </th> \n' + '<th> End Time </th> \n' + '<th> Percent Difference </th> \n' + '<th> Start Time Memory </th> \n' + '<th> End Time Memory </th> \n' + '<th> Difference </th> \n'

            for result in slnInfo.systemResultList:
                for compare in result.CompareMemoryTimeResults:
                    html += '<tr>\n'
                    html += '<td>' + result.Name + '</td>\n'
                    if(compare[2] > testingUtil.gMemoryPercentCompareMargin):
                        html += '<font color="white">' + '<td bgcolor="red">' + str(compare[0]) + '</td>' + '<td bgcolor="red">' + str(compare[1]) + '</td>' + '<td bgcolor="red">' + str(compare[2]) + '</td>' + '<td bgcolor="red">' + str(compare[3]) + '</td>' + '<td bgcolor="red">' + str(compare[4]) + '</td>' + '<td bgcolor="red">' + str(compare[5]) + '</td>' + '</font> \n'
                    else:
                        html += '<td>' + str(compare[0]) + '</td>' + '<td>' + str(compare[1]) + '</td>' + '<td>' + str(compare[2]) + '</td>' + '<td>' + str(compare[3]) + '</td>' + '<td>' + str(compare[4]) + '</td>' + '<td>' + str(compare[5]) + '</td> \n'

                    html += '</tr>\n'

            html += '</table>\n'
            html += '<br><br>'

        return html

    else:
        return ''




def skipToHTML(name, reason):
    html = '<tr>\n'
    html += '<td bgcolor="red"><font color="white">' + name + '</font></td>\n'
    html += '<td>' + reason + '</td>\n'
    html += '</tr>\n'
    return html

def getSkipsTable(slnInfo):
    if slnInfo.skippedList:
        html = '<table style="width:100%" border="1">\n'
        html += '<tr>\n'
        html += '<th colspan=\'2\'>Skipped Tests</th>'
        html += '</tr>'
        html += '<th>Test</th>\n'
        html += '<th>Reason for Skip</th>\n'
        for name, reason in slnInfo.skippedList:
            html += skipToHTML(name, reason)
        html += '</table>'
        return html
    else:
        return ''

def outputHTML(openSummary, slnInfo, pullBranch):
    html = getLowLevelTestResultsTable(slnInfo)
    html += '<br><br>'
    html += getSystemTestResultsTable(slnInfo)
    html += '<br><br>'
    html += getImageCompareResultsTable(slnInfo)
    html += '<br><br>'
    html += getMemoryCompareResultsTable(slnInfo)
    html += '<br><br>'
    html += getSkipsTable(slnInfo)
    if pullBranch:
        resultSummaryName = slnInfo.resultsDir + '\\' + slnInfo.name + '_' + pullBranch + '_TestSummary.html'
    else:
        resultSummaryName = slnInfo.resultsDir + '\\' + slnInfo.name + '_TestSummary.html'
    outfile = open(resultSummaryName, 'w')
    outfile.write(html)
    outfile.close()
    if openSummary:
        os.system("start " + resultSummaryName)
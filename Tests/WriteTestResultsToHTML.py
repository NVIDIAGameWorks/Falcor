import pprint
import os
import InternalConfig as iConfig

# Get the html end.
def get_html_begin():
    return "<!DOCTYPE html> \n" +  " <html> \n"

# Get the html end.
def get_html_end():
    return "\n </html>"


def writeErrorMessageHtml(error_message):
    html_code = ""
    html_code = html_code + get_html_begin()

    html_code = html_code + "<body>"
    html_code = html_code + error_message
    html_code = html_code + "</body>"

    html_code = html_code + get_html_end()
    return html_code


# Get the memory check table code.
def get_memory_check_table_code(tests_sets_results):
    return ["", ""]

def get_performance_check_table_code(tests_sets_results):
    return ["", ""]

def get_image_output_table_code(main_name, screen_captures, errors):
    
    ouput_table_code = ''
    ouput_table_code = '<table style="width:100%" border="1">\n'
    ouput_table_code += '<tr>\n'
    ouput_table_code += '<th colspan=\'' + str(2) + '\'>Image Compare Tests</th>\n'
    ouput_table_code += '</tr>\n'
    ouput_table_code += '<th>Test</th>\n'
    ouput_table_code += '<th>' + main_name + '</th>\n'
    
    for screen_capture in screen_captures:
        ouput_table_code += '<tr>\n'
        ouput_table_code += '<td>' + os.path.basename(screen_capture) + '</td>'
        ouput_table_code += '<td bgcolor="white"><font color="black">' + ' <br>' + '</font>'
        if screen_capture in errors.keys():
            for error_code in errors[screen_capture]:
                ouput_table_code += "<p><b> Error running test " + screen_capture + "</b>: " + str(error_code) + "<br></p>\n"
        ouput_table_code += '<a href= \'' + 'file:' + screen_capture +  '\'><img src=\'' + screen_capture + '\' title=\'reference\' alt=\'reference\' style=\'width:240px\'>' + ' </a>  '
        ouput_table_code += '</td>\n</tr>\n'
    
    ouput_table_code += '</table>\n'
    return ouput_table_code

def write_generate_references_to_html(main_name, screen_captures, errors):
    html_code = ""
    html_code = html_code + get_html_begin()

    html_code = html_code + "<body>"
    image_comparisons = get_image_output_table_code(main_name, screen_captures, errors)
    html_code = html_code + image_comparisons
    html_code = html_code + '\n <hr> \n'

    html_code = html_code + "</body>"

    html_code = html_code + get_html_end()
    return html_code

    
# Get the image comparison table code.
def get_image_comparison_table_code(screen_capture_results, errors):
    max_comparison_count = 0
    
    # Find table width
    for result_key in screen_capture_results.keys():
        for screen_capture_result in screen_capture_results[result_key]:
            max_comparison_count =  max(max_comparison_count, len(screen_capture_result))

    if max_comparison_count == 0:
        return ""
    else:
        image_comparison_table_code = '<table style="width:100%" border="1">\n'
        image_comparison_table_code += '<tr>\n'
        image_comparison_table_code += '<th colspan=\'' + str(max_comparison_count + 1) + '\'>Image Compare Tests</th>\n'
        image_comparison_table_code += '</tr>\n'
        image_comparison_table_code += '<th>Test</th>\n'
        for result_key in screen_capture_results.keys():
            image_comparison_table_code += '<th>' + result_key + '</th>\n'
    
        # map test name to result_key
        table_code_data = {}
        
        for result_key in screen_capture_results.keys():
            for test_key in screen_capture_results[result_key].keys():
                if (test_key == 'Success'):
                    continue
                
                # If zero captures, test probably failed to run. Color the test name red
                for test_result in screen_capture_results[result_key][test_key]:
                    test_name = os.path.basename(test_result['Source Filename'])
                    image_table_data = ""
                    if test_name not in table_code_data.keys():
                        table_code_data[test_name] = {}
                        table_code_data[test_name]['Success'] = True
                    table_code_data[test_name]['Success'] &= test_result['Test Passed']
                    
                    result_value = test_result['Compare Result']
                    
                    if not test_result['Test Passed']:
                        image_table_data += '<td bgcolor="red"><font color="white">' + 'Difference: ' + str(result_value) + ' <br>' + '</font>'
                    else:
                        if float(result_value) > iConfig.TestConfig['Tolerance_Lower']:
                            image_table_data += '<td bgcolor="yellow"><font color="black">' + 'Difference: ' + str(result_value) + ' <br>' + '</font>'
                        else:
                            image_table_data += '<td>' + 'Difference: ' + str(result_value)                                        
                    
                    # If this failure has an error message, add it to output
                    if test_key in errors.keys():
                        for error_code in errors[test_key]:
                            image_table_data += "<p><b> Error running test " + test_key + "</b>: " + str(error_code) + "<br></p>\n"
                            
                    # reference, local
                    if str(result_value) != 'Error':
                        image_table_data += '<a href= \'' + 'file:' + test_result["Source Filename"] +  '\'><img src=\'' + test_result['Source Filename'] + '\' title=\'source\' alt=\'source\' style=\'width:240px\'>' + ' </a>  '
                        image_table_data += '<a href= \'' + 'file:' + test_result["Reference Filename"] +  '\'><img src=\'' + test_result['Reference Filename'] + '\' title=\'reference\' alt=\'reference\' style=\'width:240px\'>' + ' </a>  '
                        image_table_data += '<a href= \'' + 'file:' + test_result['Comparison Filename'] +  '\'><img src=\'' + test_result['Comparison Filename'] + '\' title=\'comparison\' alt=\'comparison\' style=\'width:240px\'>' + ' </a>  '
                    
                    image_table_data += '<br><a href= \'' + 'file:' + os.path.dirname(test_result['Reference Filename']) +  '\'>' + 'File Location' + ' </a>  '
                    
                    image_table_data += '</td>\n'
                    table_code_data[test_name][result_key] = image_table_data
        
        for test_name in table_code_data.keys():
            image_comparison_table_code += '<tr>\n'
           
            if not table_code_data[test_name]['Success']:
                image_comparison_table_code += '<td bgcolor="red"><font color="white">' + test_name + '</font></td>'
            else:
                image_comparison_table_code += '<td>' + test_name + '</td>'
            
            for result_key in screen_capture_results.keys():
                if result_key in table_code_data[test_name].keys():
                    image_comparison_table_code += table_code_data[test_name][result_key]
                else:
                    image_comparison_table_code += '<td bgcolor="red"><font color="white">Error: No reference from ' + result_key + ' exists </font></td>\n'
            
            image_comparison_table_code += '</tr>\n'
        
        image_comparison_table_code += '</table>\n'
        return image_comparison_table_code

# Write the provided Tests Set Results to HTML and Return them.
def write_test_set_results_to_html(tests_set_results, errors):
    html_code = ""
    html_code = html_code + get_html_begin()

    html_code = html_code + "<body>"
    image_comparisons = get_image_comparison_table_code(tests_set_results, errors)
    html_code = html_code + image_comparisons
    html_code = html_code + '\n <hr> \n'

    html_code = html_code + "</body>"

    html_code = html_code + get_html_end()
    return html_code

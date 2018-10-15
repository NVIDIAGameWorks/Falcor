import pprint


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


# Get the image comparison table code.
def get_image_comparison_table_code(screen_capture_results, errors):

    max_comparison_count = 0

    # Find table width
    for screen_capture_result in screen_capture_results:
        max_comparison_count =  max(max_comparison_count, len(screen_capture_result))

    if max_comparison_count == 0:
        return ["", ""]
    else:
        image_comparison_table_code = '<table style="width:100%" border="1">\n'
        image_comparison_table_code += '<tr>\n'
        image_comparison_table_code += '<th colspan=\'' + str(max_comparison_count + 1) + '\'>Image Compare Tests</th>\n'
        image_comparison_table_code += '</tr>\n'
        image_comparison_table_code += '<th>Test</th>\n'

        image_comparison_errors_code = ""

        image_comparison_table_code += '<th>SS0' + '</th>\n'

        for screen_capture_result_key in screen_capture_results.keys():
            if (screen_capture_result_key == 'Success'):
                continue
            
            image_comparison_table_code += '<tr>\n'
            test_name = screen_capture_result_key

            # If this failure has an error message, add it to output
            if screen_capture_result_key in errors.keys():
                for error_code in errors[screen_capture_result_key]:
                    image_comparison_errors_code += "<p><b> Error running test " + test_name + "</b>: " + str(error_code) + "<br></p>\n"
            
            # If zero captures, test probably failed to run. Color the test name red
            
            for test_result in screen_capture_results[screen_capture_result_key]:
                test_name = test_result['Source Filename']
                if len(test_result) == 0:
                    image_comparison_table_code += '<td bgcolor="red"><font color="white">' + test_name + '</font></td>\n'
                    
                if len(test_result) > 0:
                    image_comparison_table_code += '<td>' + test_name + '</td>\n'
    
                    # Check if comparison was successful. It should be convertible to a number if it was
                    try:
                        result_value = float(test_result['Compare Result'])
                    except ValueError:
                        image_comparison_errors_code += "<p><b>" + test_name + " failed to compare screen capture " + "</b><br> \n"
                        image_comparison_errors_code += "<b>Source</b> : " + test_result["Source Filename"] + " <br>  <b>Reference</b> : " + test_result["Reference Filename"] + " <br> \n"
                        image_comparison_errors_code += "Please check whether the images are output correctly, whether the reference exists and whether they are the same size. <br></p>"
                        image_comparison_table_code += '<td bgcolor="red"><font color="white">Error</font></td><br>\n'
                        continue
                    
                    if not test_result['Test Passed']:
                        image_comparison_table_code += '<td bgcolor="red"><font color="white">' + str(result_value) + '</font></td>\n'
                    else:
                        if float(result_value) > 0:
                            image_comparison_table_code += '<td bgcolor="yellow"><font color="black">' + str(result_value) + '</font></td>\n'
                        else:
                            image_comparison_table_code += '<td>' + str(result_value) + '</td>\n'
    
                image_comparison_table_code += '</tr>\n'

        image_comparison_table_code += '</table>\n'
        return [image_comparison_table_code, image_comparison_errors_code]

def write_pass_test_to_html(results):
        
        return html_code
        
        
# Write the provided Tests Set Results to HTML and Return them.
def write_test_set_results_to_html(tests_set_results, errors):
    html_code = ""
    html_code = html_code + get_html_begin()

    html_code = html_code + "<body>"
    image_comparisons = get_image_comparison_table_code(tests_set_results, errors)
    html_code = html_code + image_comparisons[0]
    html_code = html_code + '\n <hr> \n'

    html_code = html_code + '\n <hr> \n'
    html_code = html_code + image_comparisons[1]

    html_code = html_code + "</body>"

    html_code = html_code + get_html_end()
    return html_code

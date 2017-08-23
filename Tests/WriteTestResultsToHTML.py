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
def get_image_comparison_table_code(tests_sets_results):

    max_image_comparison_counts = 0

    # For each test group.
    for current_test_group_result_name in tests_sets_results['Tests Groups']:
        current_test_group = tests_sets_results['Tests Groups'][current_test_group_result_name]

        if current_test_group['Enabled'] == True:

            if 'Results' in current_test_group:

                if 'Screen Capture Checks' in current_test_group['Results']:
                    screen_captures_list = current_test_group['Results']['Screen Capture Checks']

                    for screen_captures_list_index in screen_captures_list:

                        if max_image_comparison_counts < len(screen_captures_list[screen_captures_list_index].keys()):
                            max_image_comparison_counts = len(screen_captures_list[screen_captures_list_index].keys())


    if max_image_comparison_counts == 0:
        return ["", ""]

    else:

        image_comparison_table_code = '<table style="width:100%" border="1">\n'
        image_comparison_table_code += '<tr>\n'
        image_comparison_table_code += '<th colspan=\'' + str(max_image_comparison_counts + 1) + '\'>Image Compare Tests</th>\n'
        image_comparison_table_code += '</tr>\n'
        image_comparison_table_code += '<th>Test</th>\n'

        image_comparison_errors_code = ""

        for i in range (0, max_image_comparison_counts):
            image_comparison_table_code += '<th>SS' + str(i) + '</th>\n'

        for current_test_group_result_name in tests_sets_results['Tests Groups']:
            current_test_group = tests_sets_results['Tests Groups'][current_test_group_result_name]

            # Check if the current test group is enabled.
            if current_test_group['Enabled'] == True:

                if 'Results' in current_test_group:

                    #
                    if 'Screen Capture Checks' in current_test_group['Results']:
                        screen_captures_list = current_test_group['Results']['Screen Capture Checks']

                        for screen_captures_list_index in screen_captures_list:

                            # Construct the list of captures.
                            if(len(screen_captures_list[screen_captures_list_index].keys()) > 0):
                                image_comparison_table_code += '<tr>\n'
                                image_comparison_table_code += '<td>' + current_test_group_result_name + '_' + str(screen_captures_list_index) + '</td>\n'

                                #
                                for screen_capture_checks_index in screen_captures_list[screen_captures_list_index]:
                                    screen_capture_compare_result = screen_captures_list[screen_captures_list_index][screen_capture_checks_index]
                                    print screen_capture_compare_result
                                    result_value_str = screen_capture_compare_result["Compare Result"]

                                    # Check if this was a comparison.
                                    try:
                                        result_value = float(result_value_str)

                                        if not screen_capture_compare_result['Test Passed']:
                                            image_comparison_table_code += '<td bgcolor="red"><font color="white">' + str(result_value) + '</font></td>\n'
                                        else:
                                            image_comparison_table_code += '<td>' + str(result_value) + '</td>\n'

                                    except:
                                        image_comparison_errors_code = "<p> " + image_comparison_errors_code + "" + current_test_group_result_name + '_' + str(screen_captures_list_index) + " failed to compare screen capture " + str(screen_capture_checks_index) + " <br> \n"
                                        image_comparison_errors_code = image_comparison_errors_code + "Source : " + screen_capture_compare_result["Source Filename"] + " <br>  Reference : " + screen_capture_compare_result["Reference Filename"] + " <br> \n"
                                        image_comparison_errors_code = image_comparison_errors_code + "Please check whether the images are output correctly, whether the reference exists and whether they are the same size. <br> "
                                        image_comparison_errors_code = image_comparison_errors_code + "Actually, just do the references manually. <br> </p>"
                                        image_comparison_table_code += '<td bgcolor="red"><font color="white">' + str(-1) + '</font></td>\n'
                                        continue


                                image_comparison_table_code += '</tr>\n'


        image_comparison_table_code += '</table>\n'
        return [image_comparison_table_code, image_comparison_errors_code]






# Write the provided Tests Set Results to HTML and Return them.
def write_test_set_results_to_html(tests_set_results):

    html_code = ""

    html_code = html_code + get_html_begin()

    html_code = html_code + "<body>"
    if tests_set_results['Tests Set Error Status'] is True:
        html_code = html_code + '<p>' + tests_set_results['Tests Set Error Message'] + '</p>'
    else:
        image_comparisons = get_image_comparison_table_code(tests_set_results)
        html_code = html_code + image_comparisons[0]
        html_code = html_code + '\n <hr> \n'


        html_code = html_code + '\n <hr> \n'
        html_code = html_code + image_comparisons[1]

    html_code = html_code + "</body>"

    html_code = html_code + get_html_end()
    return html_code
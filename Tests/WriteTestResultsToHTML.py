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

    # Find table width
    for current_test_group_name in tests_sets_results['Tests Groups']:
        current_test_group = tests_sets_results['Tests Groups'][current_test_group_name]

        if current_test_group['Enabled'] == True:
            if 'Results' in current_test_group:
                if 'Screen Capture Checks' in current_test_group['Results']:
                    screen_captures_list_all = current_test_group['Results']['Screen Capture Checks']
                    for screen_captures_list in screen_captures_list_all:
                        count = max(len(screen_captures_list['Frame Screen Captures']), len(screen_captures_list['Time Screen Captures']))
                        screen_captures_list['Capture Count'] = count
                        max_image_comparison_counts = max(max_image_comparison_counts, count)

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

        for current_test_group_name in tests_sets_results['Tests Groups']:
            current_test_group = tests_sets_results['Tests Groups'][current_test_group_name]

            # Check if the current test group is enabled.
            if current_test_group['Enabled'] == True:
                if 'Results' in current_test_group:
                    if 'Screen Capture Checks' in current_test_group['Results']:
                        screen_captures_list = current_test_group['Results']['Screen Capture Checks']

                        # Construct the list of captures.

                        # For every test run (every time executable is ran with arguments)
                        for test_index, test_captures in enumerate(screen_captures_list):
                            image_comparison_table_code += '<tr>\n'
                            test_name = current_test_group_name + '_' + str(test_index)

                            # If zero captures, test probably failed to run. Color the test name red
                            if test_captures['Capture Count'] == 0:
                                image_comparison_table_code += '<td bgcolor="red"><font color="white">' + test_name + '</font></td>\n'

                                # If this failure has an error message, add it to output
                                if 'Errors' in current_test_group['Results'] and test_index in current_test_group['Results']['Errors']:
                                    image_comparison_errors_code += "<p><b> Error running test " + test_name + "</b>: " + str(current_test_group['Results']['Errors'][test_index]) + "<br></p>\n"

                            if test_captures['Capture Count'] > 0:
                                image_comparison_table_code += '<td>' + test_name + '</td>\n'

                                # Get the frame or time capture list, whichever one has contents
                                screen_capture_types = ['Frame Screen Captures', 'Time Screen Captures']
                                for capture_type in screen_capture_types:

                                    # For each single capture comparison result
                                    for capture_index, capture_result in enumerate(test_captures[capture_type]):

                                        # Check if comparison was successful. It should be convertible to a number if it was
                                        try:
                                            result_value = float(capture_result['Compare Result'])
                                        except ValueError:
                                            image_comparison_errors_code += "<p><b>" + test_name + " failed to compare screen capture " + str(capture_index) + "</b><br> \n"
                                            image_comparison_errors_code += "<b>Source</b> : " + capture_result["Source Filename"] + " <br>  <b>Reference</b> : " + capture_result["Reference Filename"] + " <br> \n"
                                            image_comparison_errors_code += "Please check whether the images are output correctly, whether the reference exists and whether they are the same size. <br></p>"
                                            image_comparison_table_code += '<td bgcolor="red"><font color="white">Error</font></td>\n'
                                            continue

                                        if not capture_result['Test Passed']:
                                            image_comparison_table_code += '<td bgcolor="red"><font color="white">' + str(result_value) + '</font></td>\n'
                                        else:
                                            image_comparison_table_code += '<td>' + str(result_value) + '</td>\n'

                            image_comparison_table_code += '</tr>\n'

        image_comparison_table_code += '</table>\n'
        return [image_comparison_table_code, image_comparison_errors_code]


# Write the provided Tests Set Results to HTML and Return them.
def write_test_set_results_to_html(tests_set_results):

    html_code = ""

    html_code = html_code + get_html_begin()

    html_code = html_code + "<body>"
    if tests_set_results['Error'] is not None:
        html_code = html_code + '<p>' + tests_set_results['Error'] + '</p>'
    else:
        image_comparisons = get_image_comparison_table_code(tests_set_results)
        html_code = html_code + image_comparisons[0]
        html_code = html_code + '\n <hr> \n'


        html_code = html_code + '\n <hr> \n'
        html_code = html_code + image_comparisons[1]

    html_code = html_code + "</body>"

    html_code = html_code + get_html_end()
    return html_code

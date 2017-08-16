import pprint



# Get the html end.
def get_html_begin():
    return "<!DOCTYPE html> \n" +  " <html> \n"

# Get the html end.
def get_html_end():
    return "\n </html>"



# Get the image comparison table code.
def get_image_comparison_table_code(tests_sets_results):

    max_image_comparison_counts = 0

    # For each test group.
    for current_test_group_result_name in tests_sets_results['Tests Groups']:
        current_test_group = tests_sets_results['Tests Groups'][current_test_group_result_name]

        if current_test_group['Enabled'] == 'True':

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
        image_comparison_table_code += '<th>IMG TEST</th>\n'

        image_comparison_errors_code = ""
        
        for i in range (0, max_image_comparison_counts):
            image_comparison_table_code += '<th>SS' + str(i) + '</th>\n'
        
        for current_test_group_result_name in tests_sets_results['Tests Groups']:
            current_test_group = tests_sets_results['Tests Groups'][current_test_group_result_name]

            if current_test_group['Enabled'] == 'True':

                if 'Results' in current_test_group:
                
                    if 'Screen Capture Checks' in current_test_group['Results']:
                        screen_captures_list = current_test_group['Results']['Screen Capture Checks']
    
                        for screen_captures_list_index in screen_captures_list:

                            if(len(screen_captures_list[screen_captures_list_index].keys()) > 0):
                                image_comparison_table_code += '<tr>\n'
                                image_comparison_table_code += '<td>' + current_test_group_result_name + '_' + str(screen_captures_list_index) + '</td>\n'
                        
                                for current_screen_capture_checks_index in screen_captures_list[screen_captures_list_index]:

                                    if screen_captures_list[screen_captures_list_index][current_screen_capture_checks_index][0] != 0:
                                        image_comparison_table_code += '<td bgcolor="red"><font color="white">' + str(-1) + '</font></td>\n'
                                        image_comparison_errors_code += '<p>' + screen_captures_list[screen_captures_list_index][current_screen_capture_checks_index][1] + '</p>' 
                                        continue
                            
                                    compare_value = screen_captures_list[screen_captures_list_index][current_screen_capture_checks_index][1]

                                    if float(compare_value) > 0.0:
                                        image_comparison_table_code += '<td bgcolor="red"><font color="white">' + str(compare_value) + '</font></td>\n'

                                    else:
                                        image_comparison_table_code += '<td>' + str(compare_value) + '</td>\n'

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

    print html_code
    return html_code
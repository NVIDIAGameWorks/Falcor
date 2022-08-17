% rebase('base', title='Test: ' + test['name'])

% include('snippets/nav', nav=nav)

<h4>Test: {{test['name']}}</h4>

<div class="properties">
    <div class="property">
        <div class="property-field">Duration</div>
        <div class="property-value">{{format_duration(test['duration'])}}</div>
    </div>
    <div class="property">
        <div class="property-field">References</div>
        <div class="property-value">{{ref_dir}}</div>
    </div>
    <div class="property">
        <div class="property-field">Result</div>
        <div class="property-value">
            % include('snippets/result', result=test['result'])
        </div>
    </div>
</div>

<div class="divider"></div>
<h5>Images</h5>

% if len(test['images']) == 0:
<p>No images found.</p>
% else:
% include('snippets/stats', stats=stats)
<table class="table table-striped table-hover">
    <thead>
        <tr>
            <th>Image</th>
            <th>Error</th>
            <th>Tolerance</th>
            <th>Result</th>
        </tr>
    </thead>
    <tbody>
    % for image in test['images']:
        <tr class="c-hand" onclick="window.location='/{{run_dir}}/{{test_dir}}?action=compare&image={{image['name']}}';">
            <td>{{image['name']}}</td>
            <td>{{image['error']}}</td>
            <td>{{image['tolerance']}}</td>
            <td>
                % include('snippets/result', result='PASSED' if image['success'] else 'FAILED')
            </td>
        </tr>
        % end
    </tbody>
</table>
% end

% if test['messages'] != []:
<div class="divider"></div>
<h5>Messages</h5>
    % for message in test['messages']:
    <span>{{message}}</span><br>
    % end
% end

% if 'log' in test:
<div class="divider"></div>
<h5>Log</h5>
<samp style="white-space: pre-wrap; font-size: 0.8em">{{test['log']}}</samp s>
% end

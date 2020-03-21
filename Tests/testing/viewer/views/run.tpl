% rebase('base', title='Run: ' + run['run_dir'])

% include('snippets/nav', nav=nav)

<h4>Run: {{run['run_dir']}}</h4>

<div class="properties">
    <div class="property">
        <div class="property-field">Date</div>
        <div class="property-value">{{format_date(run['date'])}}</div>
    </div>
    % for tag in run_tags:
    <div class="property">
        <div class="property-field">{{tag_titles[tag]}}</div>
        <div class="property-value">{{run['run_tags'][tag]}}</div>
    </div>
    % end
    <div class="property">
        <div class="property-field">Duration</div>
        <div class="property-value">{{format_duration(run['duration'])}}</div>
    </div>
    <div class="property">
        <div class="property-field">Result</div>
        <div class="property-value">
            % include('snippets/result', result=run['result'])
        </div>
    </div>
</div>

<div class="divider"></div>
<h5>Tests</h5>

% if len(run['tests']) == 0:
<p>No tests found.</p>
% else:
% include('snippets/stats', stats=stats)
<table class="table table-striped table-hover">
    <thead>
        <tr>
            <th>Test</th>
            <th>Images</th>
            <th>Messages</th>
            <th>Duration</th>
            <th>Result</th>
        </tr>
    </thead>
    <tbody>
        % for test in run['tests']:
        <tr class="c-hand" onclick="window.location='/{{run_dir}}/{{test['name']}}';">
            <td>{{test['name']}}</td>
            <td>{{len(test['images'])}}</td>
            <td>
                % for message in test['messages']:
                    <span>{{message}}</span><br>
                % end
            </td>
            <td>{{format_duration(test['duration'])}}</td>
            <td>
                % include('snippets/result', result=test['result'])
            </td>
        </tr>
        % end
    </tbody>
</table>
%end

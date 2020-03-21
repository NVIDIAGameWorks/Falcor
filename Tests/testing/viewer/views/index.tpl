% rebase('base', title='Home')

% include('snippets/nav', nav=nav)

<h4>Home</h4>

<div class="properties">
    <div class="property">
        <div class="property-field">Hostname</div>
        <div class="property-value">{{hostname}}</div>
    </div>
    <div class="property">
        <div class="property-field">Location</div>
        <div class="property-value">{{result_dir}}</div>
    </div>
</div>

<div class="divider"></div>
<h5>Runs</h5>

% if len(runs) == 0:
<p>No runs found.</p>
% else:
<table class="table table-striped table-hover">
    <thead>
        <tr>
            <th>Run</th>
            <th>Date</th>
            % for tag in run_tags:
            <th>{{tag_titles[tag]}}</th>
            % end
            <th>Duration</th>
            <th>Result</th>
        </tr>
    </thead>
    <tbody>
        % for run in runs:
        <tr class="c-hand" onclick="window.location='{{run['run_dir']}}';">
            <td>{{run['run_dir']}}</td>
            <td>{{format_date(run['date'])}}</td>
            % for tag in run_tags:
            <td>{{run['run_tags'][tag]}}</td>
            % end
            <td>{{format_duration(run['duration'])}}</td>
            <td>
                % include('snippets/result', result=run['result'])
            </td>
        </tr>
        % end
    </tbody>
</table>
% end

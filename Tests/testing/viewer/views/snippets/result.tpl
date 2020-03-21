
% if result == 'PASSED':
<span class="label label-success">PASSED</span>
% elif result == 'FAILED':
<span class="label label-error">FAILED</span>
% elif result == 'SKIPPED':
<span class="label label-warning">SKIPPED</span>
% else:
<span class="label">{{result}}</span>
% end

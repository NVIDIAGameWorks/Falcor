% if len(stats) > 0:
<div class="bar">
    % for item in stats:
    <div class="bar-item tooltip" data-tooltip="{{item['title']}}" style="width:{{item['percentage']}}%;background:{{item['color']}};">{{item['percentage']}}%</div>
    % end
</div>
% end

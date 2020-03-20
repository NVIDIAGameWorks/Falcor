<div class="divider"></div>
<ul class="breadcrumb">
    % for item in nav:
    <li class="breadcrumb-item">
        % if 'link' in item:
        <a href="{{item['link']}}">{{item['title']}}</a>
        % else:
        {{item['title']}}
        % end
    </li>
    % end
</ul>

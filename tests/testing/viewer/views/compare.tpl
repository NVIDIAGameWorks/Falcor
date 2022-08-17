% rebase('base', title='Compare: ' + image)

<div id="viewer" style="width: 100%; height: 600px;"></div>
<script>
    const jeri_data = {{!jeri_data}};
    Jeri.renderViewer(document.getElementById('viewer'), jeri_data);
    window.addEventListener('load', function() {
        document.getElementById('viewer').childNodes[0].focus();
    });
</script>

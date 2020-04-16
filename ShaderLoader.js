function ShaderLoader(vertex_url,fragment_url,onLoad,onProgress,onError) {
    
    var vertex_loader = new THREE.FileLoader(THREE.DefaultLoadingManager);
        vertex_loader.setResponseType('text');
        
        vertex_loader.load(vertex_url,function(vertex_text) {
        var fragment_loader = new THREE.FileLoader(THREE.DefaultLoaderManager);
            fragment_loader.setResponseType('text');
            fragment_loader.load(fragment_url,function(fragment_text) {
                onLoad(vertex_text,fragment_text);
                });
        },onProgress,onError);
} 

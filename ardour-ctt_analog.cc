@prefix doap: <http://usefulinc.com/ns/doap#> .
@prefix lv2:  <http://lv2plug.in/ns/lv2core#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ui:   <http://lv2plug.in/ns/extensions/ui#> .

<http://github.com/americosimoes/ctt>
    a lv2:Plugin ;
    lv2:requiredFeature <http://lv2plug.in/ns/ext/urid#map> ;
    
    doap:name "CTT - Riemann Zero Processor" ;
    doap:developer [
        doap:name "Americo Simoes" ;
        doap:email "americo@example.com"  # Replace with your email
    ] ;
    doap:license <http://opensource.org/licenses/isc-license> ;
    doap:description "Process audio through the lens of Riemann zeros - rich, warm spectral processing" ;
    doap:created "2026-02-17" ;
    lv2:optionalFeature lv2:hardRTCapable ;
    
    # Ports
    lv2:port [
        a lv2:InputPort , lv2:AudioPort ;
        lv2:index 0 ;
        lv2:symbol "input" ;
        lv2:name "Input" ;
        lv2:default 0.0 ;
    ] ;
    lv2:port [
        a lv2:OutputPort , lv2:AudioPort ;
        lv2:index 1 ;
        lv2:symbol "output" ;
        lv2:name "Output" ;
        lv2:default 0.0 ;
    ] ;
    lv2:port [
        a lv2:InputPort , lv2:ControlPort ;
        lv2:index 2 ;
        lv2:symbol "mix" ;
        lv2:name "Mix" ;
        lv2:default 1.0 ;
        lv2:minimum 0.0 ;
        lv2:maximum 1.0 ;
        lv2:scalePoint [ rdfs:label "Dry" ; lv2:value 0.0 ] ;
        lv2:scalePoint [ rdfs:label "Wet" ; lv2:value 1.0 ] ;
    ] ;
    lv2:port [
        a lv2:InputPort , lv2:ControlPort ;
        lv2:index 3 ;
        lv2:symbol "warmth" ;
        lv2:name "Bass Warmth" ;
        lv2:default 0.3 ;
        lv2:minimum 0.0 ;
        lv2:maximum 1.0 ;
    ] ;
    lv2:port [
        a lv2:InputPort , lv2:ControlPort ;
        lv2:index 4 ;
        lv2:symbol "resonance" ;
        lv2:name "Riemann Resonance" ;
        lv2:default 0.2 ;
        lv2:minimum 0.0 ;
        lv2:maximum 1.0 ;
    ] .

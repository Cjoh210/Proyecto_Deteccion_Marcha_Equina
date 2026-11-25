package com.cjb.eqMarc.controlador;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import com.cjb.eqMarc.servicio.*;
@RestController
@RequestMapping("/api")

public class predControl {

    private final predService servicio;

    public predControl(predService servicio){
        this.servicio = servicio;
    }

    @GetMapping("/salud")
    public Map<String, String>salud(){
        return Map.of("estado", "ok", "servicio", "API de detección de cojera equina");
    }

    @PostMapping("/prediccion")
    public ResponseEntity<?> predecir(@RequestBody Map<String, Object> solicitud) {
        if (!solicitud.containsKey("zancadas")){
            return ResponseEntity.ok(servicio.prececir(solicitud));
        }
        try {
            return ResponseEntity.ok(servicio.prececir(solicitud));
        } catch (IllegalStateException e){
            return ResponseEntity.status(502).body(
                    "Error"+e.getMessage()
            );
        }
    }
    @GetMapping("/predicciones")
    public List<predService.Registro> historial(){
        return servicio.historial();
    }
}

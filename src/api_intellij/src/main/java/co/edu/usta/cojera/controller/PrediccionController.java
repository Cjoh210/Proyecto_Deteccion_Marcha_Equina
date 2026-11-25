package co.edu.usta.cojera.controller;

import co.edu.usta.cojera.service.PrediccionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClientException;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class PrediccionController {

    private final PrediccionService servicio;

    public PrediccionController(PrediccionService servicio) {
        this.servicio = servicio;
    }

    /** Comprueba que la API está encendida. */
    @GetMapping("/salud")
    public Map<String, String> salud() {
        return Map.of("estado", "ok", "servicio", "API de detección de cojera equina");
    }

    /** Recibe las métricas de las zancadas y devuelve la probabilidad de cojera. */
    @PostMapping("/prediccion")
    public ResponseEntity<?> predecir(@RequestBody Map<String, Object> solicitud) {
        if (!solicitud.containsKey("zancadas")) {
            return ResponseEntity.badRequest().body(Map.of("error", "El cuerpo debe incluir la lista 'zancadas'"));
        }
        try {
            return ResponseEntity.ok(servicio.predecir(solicitud));
        } catch (IllegalStateException | RestClientException e) {
            return ResponseEntity.status(502).body(Map.of("error", "No se pudo consultar el modelo en AWS: " + e.getMessage()));
        }
    }

    /** Lista las predicciones hechas desde que se encendió la API. */
    @GetMapping("/predicciones")
    public List<PrediccionService.Registro> historial() {
        return servicio.historial();
    }
}

package co.edu.usta.cojera.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class PrediccionService {

    /** Una predicción ya hecha, para consultarla después. */
    public record Registro(String idCaballo, int zancadas, double probabilidad, boolean cojo, String fecha) {}

    private final RestClient cliente = RestClient.create();
    private final List<Registro> historial = new ArrayList<>();

    @Value("${aws.api.url}")
    private String urlAws;

    /** Envía las zancadas al modelo en AWS y guarda la respuesta en el historial. */
    @SuppressWarnings("unchecked")
    public Map<String, Object> predecir(Map<String, Object> solicitud) {
        if (urlAws == null || urlAws.isBlank()) {
            throw new IllegalStateException("Falta configurar aws.api.url");
        }

        Map<String, Object> respuesta = cliente.post()
                .uri(urlAws)
                .contentType(MediaType.APPLICATION_JSON)
                .body(solicitud)
                .retrieve()
                .body(Map.class);

        historial.add(new Registro(
                String.valueOf(respuesta.get("id_caballo")),
                ((Number) respuesta.get("n_zancadas")).intValue(),
                ((Number) respuesta.get("probabilidad_cojera")).doubleValue(),
                (Boolean) respuesta.get("cojo"),
                LocalDateTime.now().withNano(0).toString()));
        return respuesta;
    }

    public List<Registro> historial() {
        return historial;
    }
}

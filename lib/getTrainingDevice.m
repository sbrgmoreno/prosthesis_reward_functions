function useDevice = getTrainingDevice()
%GETTRAININGDEVICE Selecciona GPU si existe; si no, usa CPU.

    useDevice = "cpu";

    try
        hasPCT = license('test','Distrib_Computing_Toolbox');
        gpuCount = gpuDeviceCount("available");

        if hasPCT && gpuCount > 0
            g = gpuDevice(1);
            reset(g);
            useDevice = "gpu";
            fprintf("\n[GPU] Entrenamiento usando GPU: %s\n", g.Name);
        else
            fprintf("\n[CPU] No se detectó GPU disponible. Entrenamiento en CPU.\n");
        end

    catch ME
        fprintf("\n[CPU] No se pudo inicializar GPU. Entrenamiento en CPU.\n");
        fprintf("[CPU] Motivo: %s\n", ME.message);
        useDevice = "cpu";
    end
end
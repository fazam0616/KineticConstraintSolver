#version 130
in vec3 v_normal;
uniform vec3 u_lightDir;
uniform vec3 u_materialColor;
uniform float u_shininess;
out vec4 outColor;
void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_lightDir);
    float lambert = max(dot(N, L), 0.0);
    // approximate view direction towards +Z
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), max(u_shininess, 1.0));
    vec3 ambient = 0.08 * u_materialColor;
    vec3 diffuse = u_materialColor * lambert;
    vec3 specular = vec3(1.0) * spec * 0.3;
    vec3 color = ambient + diffuse + specular;
    outColor = vec4(color, 1.0);
}

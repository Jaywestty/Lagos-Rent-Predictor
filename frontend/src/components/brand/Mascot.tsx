/**
 * Adugbo, the Adugbo AI mascot: a friendly house-headed robot with a map pin on its roof.
 * Pure SVG + CSS animation (wave, blink, pin bob, float), so it is crisp at any size, weighs a few KB,
 * and goes still for anyone who prefers reduced motion. Animation classes live in index.css (m-*).
 */

const C = {
  roof: "#0f4a37",
  roofTop: "#17604a",
  limb: "#0f4a37",
  limbShade: "#0b3a2b",
  shell: "#f1ebde",
  shellEdge: "#d8ccb6",
  screen: "#0c3328",
  glow: "#8ff0c6",
  pin: "#f39a2b",
  pinDark: "#d97c12",
};

type Mood = "wave" | "think" | "idle";

export function Mascot({ size = 240, mood = "wave", className = "" }: { size?: number; mood?: Mood; className?: string }) {
  const thinking = mood === "think";
  return (
    <svg width={size} height={(size * 334) / 272} viewBox="-32 -34 272 334" role="img" aria-label="Adugbo, the Adugbo AI mascot" className={`m-root overflow-visible ${className}`}>
      {/* ground shadow, breathes with the float */}
      <ellipse className="m-shadow" cx="120" cy="290" rx="62" ry="7" fill="#0f2a22" opacity=".14" />

      <g className="m-float">
        {/* legs and shoes */}
        <rect x="90" y="226" width="24" height="44" rx="10" fill={C.limb} />
        <rect x="126" y="226" width="24" height="44" rx="10" fill={C.limb} />
        <rect x="80" y="262" width="40" height="18" rx="9" fill={C.shell} stroke={C.shellEdge} strokeWidth="2" />
        <rect x="120" y="262" width="40" height="18" rx="9" fill={C.shell} stroke={C.shellEdge} strokeWidth="2" />

        {/* arm resting on the hip */}
        <path d="M164 170 Q 194 192 172 216" fill="none" stroke={C.limb} strokeWidth="20" strokeLinecap="round" />
        <circle cx="170" cy="216" r="11" fill={C.limbShade} />

        {/* body with the logo */}
        <rect x="70" y="148" width="100" height="90" rx="32" fill={C.shell} stroke={C.shellEdge} strokeWidth="2" />
        <path d="M84 160 Q 120 150 156 160" fill="none" stroke="#fff" strokeOpacity=".55" strokeWidth="4" strokeLinecap="round" />
        <text x="110" y="216" textAnchor="middle" fontFamily="'Space Grotesk', Inter, sans-serif" fontWeight="700" fontSize="44" fill={C.limb}>a</text>
        <path d="M131 182c-5 0-9 4-9 9 0 6.5 9 15 9 15s9-8.5 9-15c0-5-4-9-9-9z" fill={C.pin} />
        <circle cx="131" cy="191" r="3.2" fill="#fff" />

        {/* head */}
        <g className="m-head">
          <rect x="50" y="56" width="140" height="106" rx="36" fill={C.shell} stroke={C.shellEdge} strokeWidth="2" />
          <rect x="44" y="96" width="10" height="28" rx="5" fill={C.limb} />
          <rect x="186" y="96" width="10" height="28" rx="5" fill={C.limb} />
          <rect x="66" y="76" width="108" height="70" rx="26" fill={C.screen} />
          <rect x="74" y="82" width="40" height="6" rx="3" fill="#fff" opacity=".08" />
          {thinking ? (
            <g className="m-look">
              <circle cx="100" cy="106" r="6" fill={C.glow} />
              <circle cx="142" cy="106" r="6" fill={C.glow} />
              <path d="M110 126 q10 4 20 0" fill="none" stroke={C.glow} strokeWidth="5" strokeLinecap="round" />
            </g>
          ) : (
            <g className="m-eyes">
              <path d="M88 108 q11 -14 22 0" fill="none" stroke={C.glow} strokeWidth="6" strokeLinecap="round" />
              <path d="M130 108 q11 -14 22 0" fill="none" stroke={C.glow} strokeWidth="6" strokeLinecap="round" />
              <path d="M108 122 q12 12 24 0" fill="none" stroke={C.glow} strokeWidth="5.5" strokeLinecap="round" />
            </g>
          )}
          {/* roof */}
          <path d="M34 78 L120 18 L206 78 L194 90 L120 38 L46 90 Z" fill={C.roof} />
          <path d="M40 74 L120 18 L200 74" fill="none" stroke={C.roofTop} strokeWidth="8" strokeLinejoin="round" strokeLinecap="round" />
          {/* map pin on the roof */}
          <g className="m-pin">
            <path d="M120 -24c-11 0-20 9-20 20 0 14 20 34 20 34s20-20 20-34c0-11-9-20-20-20z" fill={C.pin} />
            <path d="M120 -24c-11 0-20 9-20 20 0 4 1.6 8.6 4 13" fill="none" stroke="#fff" strokeOpacity=".35" strokeWidth="3" strokeLinecap="round" />
            <circle cx="120" cy="-4" r="7.5" fill="#fff" />
            <circle cx="120" cy="-4" r="3.2" fill={C.pinDark} />
          </g>
        </g>

        {/* waving arm: drawn in front of the roof, rotates from the shoulder */}
        <g className={mood === "wave" ? "m-wave" : ""}>
          <path d="M78 170 Q 40 160 24 126" fill="none" stroke={C.limb} strokeWidth="20" strokeLinecap="round" />
          <path d="M12 132 a10 10 0 0 0 20 -2" fill="none" stroke={C.pin} strokeWidth="4" strokeLinecap="round" />
          <g fill={C.limb}>
            <rect x="0" y="84" width="9" height="24" rx="4.5" transform="rotate(-16 4 96)" />
            <rect x="10" y="77" width="9" height="28" rx="4.5" transform="rotate(-5 14 91)" />
            <rect x="20" y="78" width="9" height="27" rx="4.5" transform="rotate(6 24 92)" />
            <rect x="29" y="90" width="9" height="20" rx="4.5" transform="rotate(32 33 100)" />
            <rect x="2" y="98" width="34" height="26" rx="12" />
          </g>
        </g>
        {mood === "wave" && (
          <g className="m-sparks" stroke={C.pin} strokeWidth="5" strokeLinecap="round">
            <path d="M-14 94 L -27 88" />
            <path d="M-8 72 L -17 60" />
            <path d="M10 62 L 8 47" />
          </g>
        )}
        {thinking && (
          <g className="m-dots" fill={C.limb}>
            <circle cx="200" cy="40" r="5" />
            <circle cx="216" cy="40" r="5" />
            <circle cx="232" cy="40" r="5" />
          </g>
        )}
      </g>
    </svg>
  );
}

/** Head only, for the logo, favicon-size spots and the footer. */
export function MascotHead({ size = 36, title = "Adugbo AI" }: { size?: number; title?: string }) {
  return (
    <svg width={size} height={size} viewBox="28 -26 184 190" role="img" aria-label={title}>
      <rect x="50" y="56" width="140" height="106" rx="36" fill={C.shell} stroke={C.shellEdge} strokeWidth="3" />
      <rect x="66" y="76" width="108" height="70" rx="26" fill={C.screen} />
      <path d="M88 108 q11 -14 22 0" fill="none" stroke={C.glow} strokeWidth="8" strokeLinecap="round" />
      <path d="M130 108 q11 -14 22 0" fill="none" stroke={C.glow} strokeWidth="8" strokeLinecap="round" />
      <path d="M108 124 q12 11 24 0" fill="none" stroke={C.glow} strokeWidth="7" strokeLinecap="round" />
      <path d="M34 78 L120 18 L206 78 L194 90 L120 38 L46 90 Z" fill={C.roof} />
      <path d="M120 -24c-11 0-20 9-20 20 0 14 20 34 20 34s20-20 20-34c0-11-9-20-20-20z" fill={C.pin} />
      <circle cx="120" cy="-4" r="7.5" fill="#fff" />
    </svg>
  );
}

/** Lagos skyline and the Lekki–Ikoyi link bridge, drawn softly behind the mascot. */
export function LagosScene({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 520 220" preserveAspectRatio="xMidYMax meet" aria-hidden className={className}>
      <g fill="var(--scene)" opacity=".9">
        <rect x="18" y="120" width="34" height="100" rx="3" />
        <rect x="56" y="92" width="26" height="128" rx="3" />
        <rect x="86" y="132" width="40" height="88" rx="3" />
        <rect x="130" y="108" width="22" height="112" rx="3" />
        <rect x="156" y="140" width="30" height="80" rx="3" />
        <rect x="64" y="80" width="3" height="14" />
      </g>
      {/* cable-stayed bridge */}
      <g stroke="var(--scene)" fill="none" strokeLinecap="round">
        <path d="M250 196 L 520 186" strokeWidth="9" />
        <path d="M400 192 L 410 38 L 420 190" strokeWidth="9" strokeLinejoin="round" />
        <path d="M402 120 L 418 120" strokeWidth="6" />
        {[0, 1, 2, 3, 4, 5].map((i) => <path key={`l${i}`} d={`M408 ${52 + i * 10} L ${280 + i * 18} 194`} strokeWidth="2" opacity=".85" />)}
        {[0, 1, 2, 3, 4].map((i) => <path key={`r${i}`} d={`M412 ${52 + i * 10} L ${512 - i * 16} 187`} strokeWidth="2" opacity=".85" />)}
      </g>
      <path d="M0 210 C 120 200, 220 214, 520 204 L 520 220 L 0 220 Z" fill="var(--scene)" opacity=".55" />
    </svg>
  );
}

import tailwindcss from "tailwindcss";
import autoprefixer from "autoprefixer";
import postcssPresetEnv from "postcss-preset-env";
import cssnano from "cssnano";

export default {
  plugins: [
    postcssPresetEnv({ stage: 1 }),
    tailwindcss(),
    autoprefixer(),
    ...(process.env.NODE_ENV === "production"
      ? [
          cssnano({
            preset: [
              "default",
              {
                reduceIdents: false,
                zindex: false,
                cssDeclarationSorter: false,
              },
            ],
          }),
        ]
      : []),
  ],
};

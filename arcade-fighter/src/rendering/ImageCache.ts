const cache = new Map<string, HTMLImageElement>();

export const getImage = (src: string): HTMLImageElement => {
  const existing = cache.get(src);
  if (existing) return existing;

  const img = new Image();
  img.src = src;
  cache.set(src, img);
  return img;
};

export const isImageReady = (img: HTMLImageElement): boolean => img.complete && img.naturalWidth > 0;

/**
 * Preload a set of image sources into the cache, reporting fractional progress
 * (0..1) as each one settles. An image that fails to load still counts toward
 * progress so a bad/missing asset can't stall the loading screen forever.
 */
export const preloadImages = (
  sources: string[],
  onProgress?: (fraction: number) => void
): Promise<void> => {
  const unique = Array.from(new Set(sources));
  if (unique.length === 0) {
    onProgress?.(1);
    return Promise.resolve();
  }

  let settled = 0;
  const report = () => {
    settled += 1;
    onProgress?.(settled / unique.length);
  };

  const loads = unique.map((src) => {
    const img = getImage(src);
    if (isImageReady(img)) {
      report();
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      const done = () => {
        img.removeEventListener('load', done);
        img.removeEventListener('error', done);
        report();
        resolve();
      };
      img.addEventListener('load', done);
      img.addEventListener('error', done);
    });
  });

  return Promise.all(loads).then(() => undefined);
};

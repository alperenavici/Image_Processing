// Define interface for operation parameter
export interface OperationParam {
  name: string;
  label: string;
  type: 'number' | 'range' | 'select' | 'text';
  defaultValue: any;
  min?: number;
  max?: number;
  step?: number;
  options?: { label: string; value: string | number }[];
}

// Define interface for operation configuration
export interface OperationConfig {
  name: string;
  description: string;
  requiresSecondImage?: boolean;
  params?: OperationParam[];
  defaultParams?: { [key: string]: any };
}

// Define all available operations
export const Operations: { [key: string]: OperationConfig } = {
  grayscale: {
    name: 'Gri Dönüşüm',
    description: 'Convert image to grayscale'
  },

  binary: {
    name: 'Binary Dönüşüm',
    description: 'Convert image to binary (black and white)',
    params: [
      {
        name: 'threshold',
        label: 'Threshold',
        type: 'range',
        defaultValue: 127,
        min: 0,
        max: 255,
        step: 1
      }
    ],
    defaultParams: {
      threshold: 127
    }
  },

  rotate: {
    name: 'Görüntü Döndürme',
    description: 'Rotate the image by a specific angle',
    params: [
      {
        name: 'angle',
        label: 'Angle (degrees)',
        type: 'range',
        defaultValue: 90,
        min: 0,
        max: 360,
        step: 1
      }
    ],
    defaultParams: {
      angle: 90
    }
  },

  crop: {
    name: 'Görüntü Kırpma',
    description: 'Crop the image to a specific region',
    params: [
      {
        name: 'x',
        label: 'X Start',
        type: 'number',
        defaultValue: 0
      },
      {
        name: 'y',
        label: 'Y Start',
        type: 'number',
        defaultValue: 0
      },
      {
        name: 'width',
        label: 'Width',
        type: 'number',
        defaultValue: 100
      },
      {
        name: 'height',
        label: 'Height',
        type: 'number',
        defaultValue: 100
      }
    ],
    defaultParams: {
      x: 0,
      y: 0,
      width: 100,
      height: 100
    }
  },

  zoom: {
    name: 'Görüntü Yaklaştırma/Uzaklaştırma',
    description: 'Zoom in or out of the image',
    params: [
      {
        name: 'factor',
        label: 'Zoom Factor',
        type: 'range',
        defaultValue: 1.5,
        min: 0.1,
        max: 5,
        step: 0.1
      }
    ],
    defaultParams: {
      factor: 1.5
    }
  },

  color_space: {
    name: 'Renk Uzayı Dönüşümleri',
    description: 'Convert between different color spaces',
    params: [
      {
        name: 'type',
        label: 'Conversion Type',
        type: 'select',
        defaultValue: 'rgb_to_grayscale',
        options: [
          { label: 'RGB to Grayscale', value: 'rgb_to_grayscale' },
          { label: 'Grayscale to RGB', value: 'grayscale_to_rgb' },
          { label: 'RGB to HSV', value: 'rgb_to_hsv' }
        ]
      }
    ],
    defaultParams: {
      type: 'rgb_to_grayscale'
    }
  },

  histogram: {
    name: 'Histogram Germe/Genişletme',
    description: 'Apply histogram stretching to enhance contrast'
  },

  add_images: {
    name: 'İki Resim Ekleme',
    description: 'Add two images together (pixel-wise addition)',
    requiresSecondImage: true
  },

  divide_images: {
    name: 'İki Resim Bölme',
    description: 'Divide first image by second image (pixel-wise division)',
    requiresSecondImage: true
  },

  contrast: {
    name: 'Kontrast Artırma',
    description: 'Enhance contrast by scaling around the mean',
    params: [
      {
        name: 'factor',
        label: 'Contrast Factor',
        type: 'range',
        defaultValue: 1.5,
        min: 0.5,
        max: 5,
        step: 0.1
      }
    ],
    defaultParams: {
      factor: 1.5
    }
  },

  convolution_mean: {
    name: 'Konvolüsyon İşlemi (Mean)',
    description: 'Apply mean convolution filter',
    params: [
      {
        name: 'size',
        label: 'Kernel Size',
        type: 'select',
        defaultValue: 3,
        options: [
          { label: '3x3', value: 3 },
          { label: '5x5', value: 5 },
          { label: '7x7', value: 7 }
        ]
      }
    ],
    defaultParams: {
      size: 3
    }
  },

  threshold: {
    name: 'Eşikleme İşlemi (Tek Eşikleme)',
    description: 'Apply thresholding to create a binary image',
    params: [
      {
        name: 'threshold',
        label: 'Threshold Value',
        type: 'range',
        defaultValue: 127,
        min: 0,
        max: 255,
        step: 1
      }
    ],
    defaultParams: {
      threshold: 127
    }
  },

  edge_prewitt: {
    name: 'Kenar Bulma (Prewitt)',
    description: 'Apply Prewitt edge detection'
  },

  salt_pepper: {
    name: 'Gürültü Ekleme (Salt & Pepper)',
    description: 'Add salt and pepper noise to image',
    params: [
      {
        name: 'amount',
        label: 'Noise Amount',
        type: 'range',
        defaultValue: 0.05,
        min: 0.01,
        max: 0.5,
        step: 0.01
      }
    ],
    defaultParams: {
      amount: 0.05
    }
  },

  filter_mean: {
    name: 'Gürültü Temizleme (Mean)',
    description: 'Apply mean filter to remove noise',
    params: [
      {
        name: 'size',
        label: 'Kernel Size',
        type: 'select',
        defaultValue: 3,
        options: [
          { label: '3x3', value: 3 },
          { label: '5x5', value: 5 },
          { label: '7x7', value: 7 }
        ]
      }
    ],
    defaultParams: {
      size: 3
    }
  },

  filter_median: {
    name: 'Gürültü Temizleme (Median)',
    description: 'Apply median filter to remove noise',
    params: [
      {
        name: 'size',
        label: 'Kernel Size',
        type: 'select',
        defaultValue: 3,
        options: [
          { label: '3x3', value: 3 },
          { label: '5x5', value: 5 },
          { label: '7x7', value: 7 }
        ]
      }
    ],
    defaultParams: {
      size: 3
    }
  },

  unsharp: {
    name: 'Görüntüye Filtre Uygulanması (Unsharp)',
    description: 'Apply unsharp mask filter to sharpen the image',
    params: [
      {
        name: 'strength',
        label: 'Strength',
        type: 'range',
        defaultValue: 1.0,
        min: 0.1,
        max: 5.0,
        step: 0.1
      }
    ],
    defaultParams: {
      strength: 1.0
    }
  },

  morphology_erosion: {
    name: 'Morfolojik İşlem (Aşınma)',
    description: 'Beyaz bölgeleri aşındırır, daraltır. Küçük beyaz noktaları kaldırır.',
    params: [
      {
        name: 'kernel_size',
        label: 'Aşındırma Büyüklüğü',
        type: 'select',
        defaultValue: 3,
        options: [
          { label: 'Az (3x3)', value: 3 },
          { label: 'Orta (5x5)', value: 5 },
          { label: 'Çok (7x7)', value: 7 }
        ]
      }
    ],
    defaultParams: {
      kernel_size: 3
    }
  },

  morphology_dilation: {
    name: 'Morfolojik İşlem (Genişleme)',
    description: 'Beyaz bölgeleri genişletir, büyütür. Küçük boşlukları doldurur.',
    params: [
      {
        name: 'kernel_size',
        label: 'Genişleme Büyüklüğü',
        type: 'select',
        defaultValue: 3,
        options: [
          { label: 'Az (3x3)', value: 3 },
          { label: 'Orta (5x5)', value: 5 },
          { label: 'Çok (7x7)', value: 7 }
        ]
      }
    ],
    defaultParams: {
      kernel_size: 3
    }
  },

  morphology_opening: {
    name: 'Morfolojik İşlem (Açma)',
    description: 'İlk aşındırma, sonra genişletme uygular. Küçük beyaz noktaları kaldırır.',
    params: [
      {
        name: 'kernel_size',
        label: 'İşlem Büyüklüğü',
        type: 'select',
        defaultValue: 3,
        options: [
          { label: 'Az (3x3)', value: 3 },
          { label: 'Orta (5x5)', value: 5 },
          { label: 'Çok (7x7)', value: 7 }
        ]
      }
    ],
    defaultParams: {
      kernel_size: 3
    }
  },

  morphology_closing: {
    name: 'Morfolojik İşlem (Kapama)',
    description: 'İlk genişletme, sonra aşındırma uygular. Küçük siyah delikleri kapatır.',
    params: [
      {
        name: 'kernel_size',
        label: 'İşlem Büyüklüğü',
        type: 'select',
        defaultValue: 3,
        options: [
          { label: 'Az (3x3)', value: 3 },
          { label: 'Orta (5x5)', value: 5 },
          { label: 'Çok (7x7)', value: 7 }
        ]
      }
    ],
    defaultParams: {
      kernel_size: 3
    }
  }
}; 
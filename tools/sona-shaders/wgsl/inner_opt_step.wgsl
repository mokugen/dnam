struct type_5_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: f32,
    member_5_: f32,
    member_6_: f32,
    member_7_: u32,
}

struct type_6_ {
    member: type_5_,
}

struct type_9_ {
    member: array<f32>,
}

struct type_13_ {
    member: array<vec2<f32>>,
}

struct type_17_ {
    member: array<vec4<f32>>,
}

struct type_30_ {
    member: u32,
    member_1_: u32,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
@group(0) @binding(0) 
var<uniform> uniforms: type_6_;
@group(0) @binding(1) 
var<storage, read_write> theta: type_9_;
@group(0) @binding(2) 
var<storage> grad: type_9_;
@group(0) @binding(3) 
var<storage> param_bounds: type_13_;
@group(0) @binding(4) 
var<storage, read_write> lbfgs_s_y: type_17_;
@group(0) @binding(5) 
var<storage, read_write> search_dir: type_9_;
var<workgroup> dot_scratch: array<f32, 256>;
var<workgroup> scratch_q: array<f32, 256>;
var<workgroup> lbfgs_scalars: array<f32, 32>;

fn function_() {
    var phi_2962_: f32;
    var phi_2963_: bool;
    var phi_2980_: u32;
    var phi_3001_: u32;
    var phi_3020_: bool;
    var phi_3036_: u32;
    var phi_3054_: bool;
    var phi_3070_: u32;
    var phi_3250_: bool;
    var phi_3229_: bool;
    var phi_3213_: bool;
    var phi_3235_: f32;
    var phi_3193_: bool;
    var phi_3236_: f32;
    var phi_3149_: bool;
    var phi_3155_: f32;
    var phi_3157_: f32;
    var phi_3158_: bool;
    var phi_3172_: bool;
    var phi_3178_: f32;
    var phi_3237_: f32;
    var phi_3256_: f32;
    var phi_3267_: u32;
    var phi_2981_: u32;
    var phi_3814_: bool;
    var phi_148_: f32;
    var phi_149_: bool;
    var phi_172_: f32;
    var phi_173_: bool;
    var phi_202_: u32;
    var phi_205_: f32;
    var phi_207_: f32;
    var phi_209_: f32;
    var phi_227_: u32;
    var phi_245_: bool;
    var phi_261_: u32;
    var phi_279_: bool;
    var phi_297_: u32;
    var phi_329_: bool;
    var phi_346_: bool;
    var phi_203_: u32;
    var phi_206_: f32;
    var phi_208_: f32;
    var phi_210_: f32;
    var phi_3859_: bool;
    var local: f32;
    var phi_369_: bool;
    var phi_393_: u32;
    var phi_394_: u32;
    var phi_3884_: bool;
    var phi_441_: bool;
    var local_1_: f32;
    var phi_457_: bool;
    var local_2_: f32;
    var phi_473_: bool;
    var phi_503_: u32;
    var phi_504_: u32;
    var phi_3892_: bool;
    var phi_564_: bool;
    var phi_582_: bool;
    var phi_594_: u32;
    var phi_597_: u32;
    var phi_595_: u32;
    var phi_598_: u32;
    var phi_609_: u32;
    var phi_612_: u32;
    var phi_614_: u32;
    var local_3_: u32;
    var phi_627_: u32;
    var phi_610_: u32;
    var phi_613_: u32;
    var phi_615_: u32;
    var phi_630_: u32;
    var phi_656_: bool;
    var phi_658_: bool;
    var phi_659_: bool;
    var phi_660_: bool;
    var phi_661_: bool;
    var phi_662_: bool;
    var phi_663_: bool;
    var phi_669_: u32;
    var phi_693_: u32;
    var phi_711_: bool;
    var phi_727_: u32;
    var phi_745_: bool;
    var phi_763_: u32;
    var phi_795_: bool;
    var phi_812_: bool;
    var phi_819_: vec4<f32>;
    var phi_830_: u32;
    var phi_670_: u32;
    var phi_3904_: bool;
    var phi_3943_: bool;
    var phi_3942_: bool;
    var phi_839_: u32;
    var phi_858_: u32;
    var phi_876_: bool;
    var phi_892_: u32;
    var phi_840_: u32;
    var phi_3939_: bool;
    var phi_927_: u32;
    var phi_928_: u32;
    var phi_929_: bool;
    var phi_3344_: bool;
    var phi_1020_: bool;
    var phi_3404_: bool;
    var phi_985_: bool;
    var phi_1026_: f32;
    var phi_1027_: f32;
    var phi_1028_: f32;
    var phi_1029_: bool;
    var phi_3959_: bool;
    var phi_1036_: type_30_;
    var phi_1039_: f32;
    var phi_1041_: bool;
    var phi_1037_: type_30_;
    var phi_1064_: type_30_;
    var phi_1081_: u32;
    var phi_1084_: u32;
    var phi_1082_: u32;
    var phi_1085_: u32;
    var phi_1096_: u32;
    var phi_1099_: u32;
    var phi_1101_: u32;
    var local_4_: u32;
    var phi_1114_: u32;
    var phi_1097_: u32;
    var phi_1100_: u32;
    var phi_1102_: u32;
    var phi_1117_: u32;
    var phi_1119_: u32;
    var phi_1122_: f32;
    var phi_1124_: f32;
    var phi_1126_: f32;
    var phi_1146_: u32;
    var phi_1177_: u32;
    var phi_1194_: f32;
    var phi_1120_: u32;
    var phi_1123_: f32;
    var phi_1125_: f32;
    var phi_1127_: f32;
    var phi_3950_: bool;
    var local_5_: f32;
    var phi_1206_: bool;
    var local_6_: f32;
    var phi_1222_: bool;
    var phi_1252_: u32;
    var phi_1253_: u32;
    var phi_3984_: bool;
    var phi_1313_: bool;
    var phi_1331_: bool;
    var local_7_: f32;
    var phi_1350_: bool;
    var phi_1374_: u32;
    var phi_1375_: u32;
    var phi_3994_: bool;
    var phi_1422_: bool;
    var phi_4019_: bool;
    var phi_1428_: f32;
    var phi_3464_: bool;
    var phi_1524_: bool;
    var phi_3524_: bool;
    var phi_1489_: bool;
    var phi_1530_: f32;
    var phi_1531_: f32;
    var phi_1532_: f32;
    var phi_1533_: bool;
    var phi_1549_: bool;
    var phi_3584_: bool;
    var phi_1649_: bool;
    var phi_3644_: bool;
    var phi_1614_: bool;
    var phi_1655_: f32;
    var phi_1656_: f32;
    var phi_1657_: f32;
    var phi_1658_: bool;
    var phi_1674_: bool;
    var phi_1680_: f32;
    var phi_1683_: u32;
    var phi_1704_: u32;
    var phi_1723_: u32;
    var phi_1743_: bool;
    var phi_1684_: u32;
    var phi_4006_: bool;
    var phi_4055_: bool;
    var phi_1756_: f32;
    var phi_1757_: f32;
    var phi_1758_: f32;
    var phi_1759_: bool;
    var phi_4054_: bool;
    var phi_1761_: f32;
    var phi_1762_: f32;
    var phi_1763_: f32;
    var phi_1764_: bool;
    var phi_1765_: bool;
    var phi_4051_: bool;
    var phi_1040_: f32;
    var phi_1042_: bool;
    var phi_4050_: bool;
    var phi_1798_: u32;
    var phi_1817_: u32;
    var local_8_: f32;
    var phi_1836_: bool;
    var phi_1799_: u32;
    var phi_4056_: bool;
    var local_9_: f32;
    var phi_4072_: bool;
    var phi_1858_: type_30_;
    var phi_1859_: type_30_;
    var phi_1882_: type_30_;
    var phi_1901_: u32;
    var phi_1904_: u32;
    var phi_1902_: u32;
    var phi_1905_: u32;
    var phi_1916_: u32;
    var phi_1919_: u32;
    var phi_1921_: u32;
    var local_10_: u32;
    var phi_1934_: u32;
    var phi_1917_: u32;
    var phi_1920_: u32;
    var phi_1922_: u32;
    var phi_1937_: u32;
    var phi_1939_: u32;
    var phi_1942_: f32;
    var phi_1962_: u32;
    var phi_1981_: u32;
    var phi_1940_: u32;
    var phi_1943_: f32;
    var phi_4063_: bool;
    var local_11_: f32;
    var phi_2002_: bool;
    var phi_2026_: u32;
    var phi_2027_: u32;
    var phi_4079_: bool;
    var phi_2074_: bool;
    var phi_2106_: bool;
    var phi_2114_: u32;
    var phi_2135_: u32;
    var phi_2154_: u32;
    var phi_2174_: bool;
    var phi_2115_: u32;
    var phi_4087_: bool;
    var phi_4101_: bool;
    var phi_4100_: bool;
    var phi_2188_: u32;
    var phi_2191_: f32;
    var phi_2193_: f32;
    var phi_2211_: u32;
    var phi_2229_: bool;
    var phi_2245_: u32;
    var phi_2260_: f32;
    var phi_2272_: bool;
    var phi_2189_: u32;
    var phi_2192_: f32;
    var phi_2194_: f32;
    var phi_4102_: bool;
    var local_12_: f32;
    var phi_2291_: bool;
    var local_13_: f32;
    var phi_2307_: bool;
    var phi_2337_: u32;
    var phi_2338_: u32;
    var phi_4114_: bool;
    var phi_2398_: bool;
    var phi_2416_: bool;
    var phi_2438_: bool;
    var phi_2439_: bool;
    var phi_2440_: bool;
    var phi_3704_: bool;
    var phi_2531_: bool;
    var phi_3764_: bool;
    var phi_2496_: bool;
    var phi_2537_: f32;
    var phi_2538_: f32;
    var phi_2539_: f32;
    var phi_2540_: bool;
    var phi_2551_: u32;
    var phi_2570_: u32;
    var phi_2588_: bool;
    var phi_2604_: u32;
    var phi_2622_: bool;
    var phi_2638_: u32;
    var phi_2659_: u32;
    var phi_2699_: bool;
    var phi_2682_: bool;
    var phi_2705_: f32;
    var phi_2719_: u32;
    var phi_2736_: u32;
    var phi_2912_: bool;
    var phi_2891_: bool;
    var phi_2875_: bool;
    var phi_2897_: f32;
    var phi_2855_: bool;
    var phi_2898_: f32;
    var phi_2811_: bool;
    var phi_2817_: f32;
    var phi_2819_: f32;
    var phi_2820_: bool;
    var phi_2834_: bool;
    var phi_2840_: f32;
    var phi_2899_: f32;
    var phi_2918_: f32;
    var phi_2929_: u32;
    var phi_2552_: u32;
    var phi_4124_: bool;
    var local_14_: u32;
    var local_15_: u32;
    var local_16_: u32;
    var local_17_: u32;
    var local_18_: u32;
    var local_19_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e52_ = wg_id_1_;
            let _e53_ = lid_1_;
            let _e55_ = arrayLength((&theta.member));
            let _e57_ = arrayLength((&grad.member));
            let _e59_ = arrayLength((&param_bounds.member));
            let _e61_ = arrayLength((&lbfgs_s_y.member));
            let _e63_ = arrayLength((&search_dir.member));
            if (_e52_.x == 0u) {
                let _e66_ = (_e53_ & 255u);
                let _e69_ = uniforms.member.member;
                let _e72_ = uniforms.member.member_3_;
                if (_e72_ == 0u) {
                    let _e265_ = uniforms.member.member_1_;
                    let _e267_ = select(_e265_, 8u, (8u < _e265_));
                    let _e270_ = uniforms.member.member_5_;
                    if ((bitcast<u32>(_e270_) & 2139095040u) != 2139095040u) {
                        let _e274_ = (_e270_ > 0f);
                        phi_148_ = select(f32(), _e270_, _e274_);
                        phi_149_ = select(true, false, _e274_);
                    } else {
                        phi_148_ = f32();
                        phi_149_ = true;
                    }
                    let _e278_ = phi_148_;
                    let _e280_ = phi_149_;
                    let _e281_ = select(_e278_, 0f, _e280_);
                    let _e284_ = uniforms.member.member_6_;
                    if ((bitcast<u32>(_e284_) & 2139095040u) != 2139095040u) {
                        let _e288_ = (_e284_ >= 0f);
                        phi_172_ = select(f32(), _e284_, _e288_);
                        phi_173_ = select(true, false, _e288_);
                    } else {
                        phi_172_ = f32();
                        phi_173_ = true;
                    }
                    let _e292_ = phi_172_;
                    let _e294_ = phi_173_;
                    let _e295_ = select(_e292_, 0f, _e294_);
                    let _e298_ = uniforms.member.member_4_;
                    let _e305_ = uniforms.member.member_2_;
                    if (_e305_ > 0u) {
                        if (_e267_ > 0u) {
                            phi_202_ = _e66_;
                            phi_205_ = 0f;
                            phi_207_ = 0f;
                            phi_209_ = 0f;
                            loop {
                                let _e309_ = phi_202_;
                                let _e311_ = phi_205_;
                                let _e313_ = phi_207_;
                                let _e315_ = phi_209_;
                                local = _e315_;
                                local_1_ = _e313_;
                                local_2_ = _e311_;
                                let _e316_ = (_e309_ < _e69_);
                                if _e316_ {
                                    if (_e55_ == 0u) {
                                        phi_227_ = 0u;
                                    } else {
                                        let _e318_ = (_e55_ - 1u);
                                        phi_227_ = select(_e318_, _e309_, (_e309_ < _e318_));
                                    }
                                    let _e322_ = phi_227_;
                                    if (_e322_ < _e55_) {
                                    } else {
                                        phi_3859_ = true;
                                        break;
                                    }
                                    let _e326_ = theta.member[_e322_];
                                    let _e327_ = bitcast<u32>(_e326_);
                                    let _e328_ = (_e327_ & 2139095040u);
                                    if (_e328_ == 2139095040u) {
                                        phi_245_ = true;
                                    } else {
                                        phi_245_ = (_e328_ == 0u);
                                    }
                                    let _e332_ = phi_245_;
                                    if (_e57_ == 0u) {
                                        phi_261_ = 0u;
                                    } else {
                                        let _e336_ = (_e57_ - 1u);
                                        phi_261_ = select(_e336_, _e309_, (_e309_ < _e336_));
                                    }
                                    let _e340_ = phi_261_;
                                    if (_e340_ < _e57_) {
                                    } else {
                                        phi_3859_ = true;
                                        break;
                                    }
                                    let _e344_ = grad.member[_e340_];
                                    let _e345_ = bitcast<u32>(_e344_);
                                    let _e346_ = (_e345_ & 2139095040u);
                                    if (_e346_ == 2139095040u) {
                                        phi_279_ = true;
                                    } else {
                                        phi_279_ = (_e346_ == 0u);
                                    }
                                    let _e350_ = phi_279_;
                                    let _e354_ = ((_e267_ * _e69_) + _e309_);
                                    if (_e61_ == 0u) {
                                        phi_297_ = 0u;
                                    } else {
                                        let _e356_ = (_e61_ - 1u);
                                        phi_297_ = select(_e356_, _e354_, (_e354_ < _e356_));
                                    }
                                    let _e360_ = phi_297_;
                                    if (_e360_ < _e61_) {
                                    } else {
                                        phi_3859_ = true;
                                        break;
                                    }
                                    let _e365_ = lbfgs_s_y.member[_e360_].x;
                                    let _e369_ = lbfgs_s_y.member[_e360_].y;
                                    let _e371_ = bitcast<u32>((bitcast<f32>(select(_e327_, 0u, _e332_)) - _e365_));
                                    let _e372_ = (_e371_ & 2139095040u);
                                    if (_e372_ == 2139095040u) {
                                        phi_329_ = true;
                                    } else {
                                        phi_329_ = (_e372_ == 0u);
                                    }
                                    let _e376_ = phi_329_;
                                    let _e378_ = bitcast<f32>(select(_e371_, 0u, _e376_));
                                    let _e380_ = bitcast<u32>((bitcast<f32>(select(_e345_, 0u, _e350_)) - _e369_));
                                    let _e381_ = (_e380_ & 2139095040u);
                                    if (_e381_ == 2139095040u) {
                                        phi_346_ = true;
                                    } else {
                                        phi_346_ = (_e381_ == 0u);
                                    }
                                    let _e385_ = phi_346_;
                                    let _e387_ = bitcast<f32>(select(_e380_, 0u, _e385_));
                                    phi_203_ = (_e309_ + 256u);
                                    phi_206_ = (_e311_ + (_e387_ * _e387_));
                                    phi_208_ = (_e313_ + (_e378_ * _e378_));
                                    phi_210_ = (_e315_ + (_e378_ * _e387_));
                                } else {
                                    phi_203_ = u32();
                                    phi_206_ = f32();
                                    phi_208_ = f32();
                                    phi_210_ = f32();
                                }
                                let _e396_ = phi_203_;
                                let _e398_ = phi_206_;
                                let _e400_ = phi_208_;
                                let _e402_ = phi_210_;
                                phi_202_ = _e396_;
                                phi_205_ = _e398_;
                                phi_207_ = _e400_;
                                phi_209_ = _e402_;
                                phi_3859_ = false;
                                if !(_e316_) {
                                    break;
                                }
                            }
                            let _e405_ = phi_3859_;
                            if _e405_ {
                                break;
                            }
                            let _e407_ = local;
                            let _e408_ = bitcast<u32>(_e407_);
                            let _e409_ = (_e408_ & 2139095040u);
                            if (_e409_ == 2139095040u) {
                                phi_369_ = true;
                            } else {
                                phi_369_ = (_e409_ == 0u);
                            }
                            let _e413_ = phi_369_;
                            let _e417_ = select(255u, _e66_, (_e66_ < 255u));
                            let _e418_ = (_e417_ < 256u);
                            if _e418_ {
                            } else {
                                break;
                            }
                            dot_scratch[_e417_] = bitcast<f32>(select(_e408_, 0u, _e413_));
                            workgroupBarrier();
                            phi_393_ = 128u;
                            loop {
                                let _e421_ = phi_393_;
                                let _e422_ = (_e421_ > 0u);
                                if _e422_ {
                                    if (_e66_ < _e421_) {
                                        let _e424_ = (_e66_ < 256u);
                                        if _e424_ {
                                        } else {
                                            phi_3884_ = true;
                                            break;
                                        }
                                        let _e426_ = dot_scratch[_e66_];
                                        let _e427_ = (_e66_ + _e421_);
                                        if (_e427_ < 256u) {
                                        } else {
                                            phi_3884_ = true;
                                            break;
                                        }
                                        let _e430_ = dot_scratch[_e427_];
                                        if _e424_ {
                                        } else {
                                            phi_3884_ = true;
                                            break;
                                        }
                                        dot_scratch[_e66_] = (_e426_ + _e430_);
                                    }
                                    workgroupBarrier();
                                    phi_394_ = (_e421_ >> bitcast<u32>(1i));
                                } else {
                                    phi_394_ = u32();
                                }
                                let _e435_ = phi_394_;
                                phi_393_ = _e435_;
                                phi_3884_ = _e405_;
                                if !(_e422_) {
                                    break;
                                }
                            }
                            let _e438_ = phi_3884_;
                            if _e438_ {
                                break;
                            }
                            let _e440_ = dot_scratch[0];
                            let _e441_ = bitcast<u32>(_e440_);
                            let _e442_ = (_e441_ & 2139095040u);
                            if (_e442_ == 2139095040u) {
                                phi_441_ = true;
                            } else {
                                phi_441_ = (_e442_ == 0u);
                            }
                            let _e446_ = phi_441_;
                            let _e448_ = bitcast<f32>(select(_e441_, 0u, _e446_));
                            workgroupBarrier();
                            let _e450_ = local_1_;
                            let _e451_ = bitcast<u32>(_e450_);
                            let _e452_ = (_e451_ & 2139095040u);
                            if (_e452_ == 2139095040u) {
                                phi_457_ = true;
                            } else {
                                phi_457_ = (_e452_ == 0u);
                            }
                            let _e456_ = phi_457_;
                            let _e460_ = local_2_;
                            let _e461_ = bitcast<u32>(_e460_);
                            let _e462_ = (_e461_ & 2139095040u);
                            if (_e462_ == 2139095040u) {
                                phi_473_ = true;
                            } else {
                                phi_473_ = (_e462_ == 0u);
                            }
                            let _e466_ = phi_473_;
                            if _e418_ {
                            } else {
                                break;
                            }
                            scratch_q[_e417_] = bitcast<f32>(select(_e451_, 0u, _e456_));
                            if _e418_ {
                            } else {
                                break;
                            }
                            dot_scratch[_e417_] = bitcast<f32>(select(_e461_, 0u, _e466_));
                            workgroupBarrier();
                            phi_503_ = 128u;
                            loop {
                                let _e471_ = phi_503_;
                                let _e472_ = (_e471_ > 0u);
                                if _e472_ {
                                    if (_e66_ < _e471_) {
                                        let _e474_ = (_e66_ + _e471_);
                                        let _e475_ = (_e474_ < 256u);
                                        if _e475_ {
                                        } else {
                                            phi_3892_ = true;
                                            break;
                                        }
                                        let _e477_ = scratch_q[_e474_];
                                        let _e478_ = (_e66_ < 256u);
                                        if _e478_ {
                                        } else {
                                            phi_3892_ = true;
                                            break;
                                        }
                                        let _e480_ = scratch_q[_e66_];
                                        scratch_q[_e66_] = (_e480_ + _e477_);
                                        if _e475_ {
                                        } else {
                                            phi_3892_ = true;
                                            break;
                                        }
                                        let _e483_ = dot_scratch[_e474_];
                                        if _e478_ {
                                        } else {
                                            phi_3892_ = true;
                                            break;
                                        }
                                        let _e485_ = dot_scratch[_e66_];
                                        dot_scratch[_e66_] = (_e485_ + _e483_);
                                    }
                                    workgroupBarrier();
                                    phi_504_ = (_e471_ >> bitcast<u32>(1i));
                                } else {
                                    phi_504_ = u32();
                                }
                                let _e490_ = phi_504_;
                                phi_503_ = _e490_;
                                phi_3892_ = _e438_;
                                if !(_e472_) {
                                    break;
                                }
                            }
                            let _e493_ = phi_3892_;
                            if _e493_ {
                                break;
                            }
                            let _e495_ = scratch_q[0];
                            let _e496_ = bitcast<u32>(_e495_);
                            let _e497_ = (_e496_ & 2139095040u);
                            if (_e497_ == 2139095040u) {
                                phi_564_ = true;
                            } else {
                                phi_564_ = (_e497_ == 0u);
                            }
                            let _e501_ = phi_564_;
                            let _e503_ = bitcast<f32>(select(_e496_, 0u, _e501_));
                            let _e504_ = dot_scratch[0];
                            let _e505_ = bitcast<u32>(_e504_);
                            let _e506_ = (_e505_ & 2139095040u);
                            if (_e506_ == 2139095040u) {
                                phi_582_ = true;
                            } else {
                                phi_582_ = (_e506_ == 0u);
                            }
                            let _e510_ = phi_582_;
                            let _e512_ = bitcast<f32>(select(_e505_, 0u, _e510_));
                            workgroupBarrier();
                            let _e513_ = (_e305_ - 1u);
                            if (_e267_ <= 1u) {
                                phi_630_ = 0u;
                            } else {
                                phi_594_ = 0u;
                                phi_597_ = _e267_;
                                loop {
                                    let _e516_ = phi_594_;
                                    let _e518_ = phi_597_;
                                    local_14_ = _e518_;
                                    local_3_ = _e516_;
                                    let _e521_ = (_e518_ <= (_e513_ >> bitcast<u32>(1i)));
                                    if _e521_ {
                                        phi_595_ = (_e516_ + 1u);
                                        phi_598_ = (_e518_ << bitcast<u32>(1i));
                                    } else {
                                        phi_595_ = u32();
                                        phi_598_ = u32();
                                    }
                                    let _e526_ = phi_595_;
                                    let _e528_ = phi_598_;
                                    phi_594_ = _e526_;
                                    phi_597_ = _e528_;
                                    if !(_e521_) {
                                        break;
                                    }
                                }
                                phi_609_ = 0u;
                                let _e2047_ = local_14_;
                                phi_612_ = _e2047_;
                                phi_614_ = _e513_;
                                loop {
                                    let _e531_ = phi_609_;
                                    let _e533_ = phi_612_;
                                    let _e535_ = phi_614_;
                                    local_15_ = _e535_;
                                    let _e537_ = local_3_;
                                    let _e538_ = (_e531_ <= _e537_);
                                    if _e538_ {
                                        if (_e535_ >= _e533_) {
                                            phi_627_ = (_e535_ - _e533_);
                                        } else {
                                            phi_627_ = _e535_;
                                        }
                                        let _e542_ = phi_627_;
                                        phi_610_ = (_e531_ + 1u);
                                        phi_613_ = (_e533_ >> bitcast<u32>(1i));
                                        phi_615_ = _e542_;
                                    } else {
                                        phi_610_ = u32();
                                        phi_613_ = u32();
                                        phi_615_ = u32();
                                    }
                                    let _e547_ = phi_610_;
                                    let _e549_ = phi_613_;
                                    let _e551_ = phi_615_;
                                    phi_609_ = _e547_;
                                    phi_612_ = _e549_;
                                    phi_614_ = _e551_;
                                    if !(_e538_) {
                                        break;
                                    }
                                }
                                let _e2056_ = local_15_;
                                phi_630_ = _e2056_;
                            }
                            let _e554_ = phi_630_;
                            if ((bitcast<u32>(_e448_) & 2139095040u) != 2139095040u) {
                                if ((bitcast<u32>(_e503_) & 2139095040u) != 2139095040u) {
                                    if ((bitcast<u32>(_e512_) & 2139095040u) != 2139095040u) {
                                        let _e566_ = (_e448_ > (0.000000000001f * (_e503_ + _e512_)));
                                        if _e566_ {
                                            phi_656_ = (_e448_ > 0f);
                                        } else {
                                            phi_656_ = bool();
                                        }
                                        let _e569_ = phi_656_;
                                        phi_658_ = _e569_;
                                        phi_659_ = select(true, false, _e566_);
                                    } else {
                                        phi_658_ = bool();
                                        phi_659_ = true;
                                    }
                                    let _e572_ = phi_658_;
                                    let _e574_ = phi_659_;
                                    phi_660_ = _e572_;
                                    phi_661_ = _e574_;
                                } else {
                                    phi_660_ = bool();
                                    phi_661_ = true;
                                }
                                let _e576_ = phi_660_;
                                let _e578_ = phi_661_;
                                phi_662_ = _e576_;
                                phi_663_ = _e578_;
                            } else {
                                phi_662_ = bool();
                                phi_663_ = true;
                            }
                            let _e580_ = phi_662_;
                            let _e582_ = phi_663_;
                            phi_669_ = _e66_;
                            loop {
                                let _e585_ = phi_669_;
                                let _e586_ = (_e585_ < _e69_);
                                if _e586_ {
                                    let _e588_ = ((_e554_ * _e69_) + _e585_);
                                    if select(_e580_, false, _e582_) {
                                        if (_e55_ == 0u) {
                                            phi_693_ = 0u;
                                        } else {
                                            let _e590_ = (_e55_ - 1u);
                                            phi_693_ = select(_e590_, _e585_, (_e585_ < _e590_));
                                        }
                                        let _e594_ = phi_693_;
                                        if (_e594_ < _e55_) {
                                        } else {
                                            phi_3904_ = true;
                                            break;
                                        }
                                        let _e598_ = theta.member[_e594_];
                                        let _e599_ = bitcast<u32>(_e598_);
                                        let _e600_ = (_e599_ & 2139095040u);
                                        if (_e600_ == 2139095040u) {
                                            phi_711_ = true;
                                        } else {
                                            phi_711_ = (_e600_ == 0u);
                                        }
                                        let _e604_ = phi_711_;
                                        if (_e57_ == 0u) {
                                            phi_727_ = 0u;
                                        } else {
                                            let _e608_ = (_e57_ - 1u);
                                            phi_727_ = select(_e608_, _e585_, (_e585_ < _e608_));
                                        }
                                        let _e612_ = phi_727_;
                                        if (_e612_ < _e57_) {
                                        } else {
                                            phi_3904_ = true;
                                            break;
                                        }
                                        let _e616_ = grad.member[_e612_];
                                        let _e617_ = bitcast<u32>(_e616_);
                                        let _e618_ = (_e617_ & 2139095040u);
                                        if (_e618_ == 2139095040u) {
                                            phi_745_ = true;
                                        } else {
                                            phi_745_ = (_e618_ == 0u);
                                        }
                                        let _e622_ = phi_745_;
                                        let _e626_ = ((_e267_ * _e69_) + _e585_);
                                        if (_e61_ == 0u) {
                                            phi_763_ = 0u;
                                        } else {
                                            let _e628_ = (_e61_ - 1u);
                                            phi_763_ = select(_e628_, _e626_, (_e626_ < _e628_));
                                        }
                                        let _e632_ = phi_763_;
                                        if (_e632_ < _e61_) {
                                        } else {
                                            phi_3904_ = true;
                                            break;
                                        }
                                        let _e637_ = lbfgs_s_y.member[_e632_].x;
                                        let _e641_ = lbfgs_s_y.member[_e632_].y;
                                        let _e643_ = bitcast<u32>((bitcast<f32>(select(_e599_, 0u, _e604_)) - _e637_));
                                        let _e644_ = (_e643_ & 2139095040u);
                                        if (_e644_ == 2139095040u) {
                                            phi_795_ = true;
                                        } else {
                                            phi_795_ = (_e644_ == 0u);
                                        }
                                        let _e648_ = phi_795_;
                                        let _e652_ = bitcast<u32>((bitcast<f32>(select(_e617_, 0u, _e622_)) - _e641_));
                                        let _e653_ = (_e652_ & 2139095040u);
                                        if (_e653_ == 2139095040u) {
                                            phi_812_ = true;
                                        } else {
                                            phi_812_ = (_e653_ == 0u);
                                        }
                                        let _e657_ = phi_812_;
                                        phi_819_ = vec4<f32>(bitcast<f32>(select(_e643_, 0u, _e648_)), bitcast<f32>(select(_e652_, 0u, _e657_)), 0f, 0f);
                                    } else {
                                        phi_819_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e662_ = phi_819_;
                                    if (_e61_ == 0u) {
                                        phi_830_ = 0u;
                                    } else {
                                        let _e664_ = (_e61_ - 1u);
                                        phi_830_ = select(_e664_, _e588_, (_e588_ < _e664_));
                                    }
                                    let _e668_ = phi_830_;
                                    if (_e668_ < _e61_) {
                                    } else {
                                        phi_3904_ = true;
                                        break;
                                    }
                                    lbfgs_s_y.member[_e668_] = _e662_;
                                    phi_670_ = (_e585_ + 256u);
                                } else {
                                    phi_670_ = u32();
                                }
                                let _e674_ = phi_670_;
                                phi_669_ = _e674_;
                                phi_3904_ = _e493_;
                                if !(_e586_) {
                                    break;
                                }
                            }
                            let _e677_ = phi_3904_;
                            if _e677_ {
                                break;
                            }
                            workgroupBarrier();
                            phi_3943_ = _e677_;
                        } else {
                            phi_3943_ = false;
                        }
                        let _e679_ = phi_3943_;
                        phi_3942_ = _e679_;
                    } else {
                        phi_3942_ = false;
                    }
                    let _e681_ = phi_3942_;
                    phi_839_ = _e66_;
                    loop {
                        let _e683_ = phi_839_;
                        let _e684_ = (_e683_ < _e69_);
                        if _e684_ {
                            if (_e57_ == 0u) {
                                phi_858_ = 0u;
                            } else {
                                let _e686_ = (_e57_ - 1u);
                                phi_858_ = select(_e686_, _e683_, (_e683_ < _e686_));
                            }
                            let _e690_ = phi_858_;
                            if (_e690_ < _e57_) {
                            } else {
                                phi_3939_ = true;
                                break;
                            }
                            let _e694_ = grad.member[_e690_];
                            let _e695_ = bitcast<u32>(_e694_);
                            let _e696_ = (_e695_ & 2139095040u);
                            if (_e696_ == 2139095040u) {
                                phi_876_ = true;
                            } else {
                                phi_876_ = (_e696_ == 0u);
                            }
                            let _e700_ = phi_876_;
                            if (_e63_ == 0u) {
                                phi_892_ = 0u;
                            } else {
                                let _e704_ = (_e63_ - 1u);
                                phi_892_ = select(_e704_, _e683_, (_e683_ < _e704_));
                            }
                            let _e708_ = phi_892_;
                            if (_e708_ < _e63_) {
                            } else {
                                phi_3939_ = true;
                                break;
                            }
                            search_dir.member[_e708_] = bitcast<f32>(select(_e695_, 0u, _e700_));
                            phi_840_ = (_e683_ + 256u);
                        } else {
                            phi_840_ = u32();
                        }
                        let _e714_ = phi_840_;
                        phi_839_ = _e714_;
                        phi_3939_ = _e681_;
                        if !(_e684_) {
                            break;
                        }
                    }
                    let _e717_ = phi_3939_;
                    if _e717_ {
                        break;
                    }
                    let _e718_ = (_e66_ < 32u);
                    if _e718_ {
                        if _e718_ {
                        } else {
                            break;
                        }
                        lbfgs_scalars[_e66_] = 0f;
                    }
                    workgroupBarrier();
                    if (_e305_ == 0u) {
                        phi_928_ = u32();
                        phi_929_ = true;
                    } else {
                        let _e721_ = (_e267_ == 0u);
                        if _e721_ {
                            phi_927_ = u32();
                        } else {
                            phi_927_ = select(_e305_, _e267_, (_e267_ < _e305_));
                        }
                        let _e725_ = phi_927_;
                        phi_928_ = _e725_;
                        phi_929_ = _e721_;
                    }
                    let _e727_ = phi_928_;
                    let _e729_ = phi_929_;
                    let _e730_ = select(_e727_, 0u, _e729_);
                    let _e731_ = (1f + _e281_);
                    let _e734_ = ((bitcast<u32>(_e731_) & 2139095040u) != 2139095040u);
                    if _e734_ {
                        let _e735_ = (_e731_ <= 0f);
                        if _e735_ {
                            phi_1027_ = f32();
                        } else {
                            if (_e731_ < 0.00001f) {
                                let _e767_ = (select(0.000000000000000000000001f, _e731_, (_e731_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                if (_e767_ != _e767_) {
                                    phi_3404_ = true;
                                } else {
                                    phi_3404_ = (0.000001f >= _e767_);
                                }
                                let _e771_ = phi_3404_;
                                let _e772_ = select(_e767_, 0.000001f, _e771_);
                                let _e775_ = bitcast<f32>((2129859011u - bitcast<u32>(_e772_)));
                                let _e779_ = (_e775_ + (_e775_ * (1f - (_e772_ * _e775_))));
                                let _e783_ = (_e779_ + (_e779_ * (1f - (_e772_ * _e779_))));
                                let _e789_ = bitcast<u32>(((_e783_ + (_e783_ * (1f - (_e772_ * _e783_)))) * 18446744000000000000f));
                                let _e790_ = (_e789_ & 2139095040u);
                                if (_e790_ == 2139095040u) {
                                    phi_985_ = true;
                                } else {
                                    phi_985_ = (_e790_ == 0u);
                                }
                                let _e794_ = phi_985_;
                                phi_1026_ = bitcast<f32>(select(_e789_, 0u, _e794_));
                            } else {
                                if (_e731_ != _e731_) {
                                    phi_3344_ = true;
                                } else {
                                    phi_3344_ = (0.000001f >= _e731_);
                                }
                                let _e740_ = phi_3344_;
                                let _e741_ = select(_e731_, 0.000001f, _e740_);
                                let _e744_ = bitcast<f32>((2129859011u - bitcast<u32>(_e741_)));
                                let _e748_ = (_e744_ + (_e744_ * (1f - (_e741_ * _e744_))));
                                let _e752_ = (_e748_ + (_e748_ * (1f - (_e741_ * _e748_))));
                                let _e757_ = bitcast<u32>((_e752_ + (_e752_ * (1f - (_e741_ * _e752_)))));
                                let _e758_ = (_e757_ & 2139095040u);
                                if (_e758_ == 2139095040u) {
                                    phi_1020_ = true;
                                } else {
                                    phi_1020_ = (_e758_ == 0u);
                                }
                                let _e762_ = phi_1020_;
                                phi_1026_ = bitcast<f32>(select(_e757_, 0u, _e762_));
                            }
                            let _e798_ = phi_1026_;
                            phi_1027_ = _e798_;
                        }
                        let _e800_ = phi_1027_;
                        phi_1028_ = _e800_;
                        phi_1029_ = _e735_;
                    } else {
                        phi_1028_ = f32();
                        phi_1029_ = true;
                    }
                    let _e802_ = phi_1028_;
                    let _e804_ = phi_1029_;
                    let _e806_ = type_30_(0u, _e730_);
                    phi_3959_ = _e717_;
                    phi_1036_ = _e806_;
                    phi_1039_ = select(_e802_, 0f, _e804_);
                    phi_1041_ = false;
                    loop {
                        let _e808_ = phi_3959_;
                        let _e810_ = phi_1036_;
                        let _e812_ = phi_1039_;
                        let _e814_ = phi_1041_;
                        local_8_ = _e812_;
                        local_9_ = _e812_;
                        if (_e810_.member < _e810_.member_1_) {
                            phi_1037_ = type_30_((_e810_.member + 1u), _e810_.member_1_);
                            phi_1064_ = type_30_(1u, _e810_.member);
                        } else {
                            phi_1037_ = _e810_;
                            phi_1064_ = type_30_(0u, u32());
                        }
                        let _e827_ = phi_1037_;
                        let _e829_ = phi_1064_;
                        let _e833_ = (bitcast<i32>(_e829_.member) != 0i);
                        if _e833_ {
                            let _e835_ = ((_e305_ - 1u) - _e829_.member_1_);
                            if (_e267_ <= 1u) {
                                phi_1117_ = 0u;
                            } else {
                                phi_1081_ = 0u;
                                phi_1084_ = _e267_;
                                loop {
                                    let _e838_ = phi_1081_;
                                    let _e840_ = phi_1084_;
                                    local_16_ = _e840_;
                                    local_4_ = _e838_;
                                    let _e843_ = (_e840_ <= (_e835_ >> bitcast<u32>(1i)));
                                    if _e843_ {
                                        phi_1082_ = (_e838_ + 1u);
                                        phi_1085_ = (_e840_ << bitcast<u32>(1i));
                                    } else {
                                        phi_1082_ = u32();
                                        phi_1085_ = u32();
                                    }
                                    let _e848_ = phi_1082_;
                                    let _e850_ = phi_1085_;
                                    phi_1081_ = _e848_;
                                    phi_1084_ = _e850_;
                                    if !(_e843_) {
                                        break;
                                    }
                                }
                                phi_1096_ = 0u;
                                let _e2108_ = local_16_;
                                phi_1099_ = _e2108_;
                                phi_1101_ = _e835_;
                                loop {
                                    let _e853_ = phi_1096_;
                                    let _e855_ = phi_1099_;
                                    let _e857_ = phi_1101_;
                                    local_17_ = _e857_;
                                    let _e859_ = local_4_;
                                    let _e860_ = (_e853_ <= _e859_);
                                    if _e860_ {
                                        if (_e857_ >= _e855_) {
                                            phi_1114_ = (_e857_ - _e855_);
                                        } else {
                                            phi_1114_ = _e857_;
                                        }
                                        let _e864_ = phi_1114_;
                                        phi_1097_ = (_e853_ + 1u);
                                        phi_1100_ = (_e855_ >> bitcast<u32>(1i));
                                        phi_1102_ = _e864_;
                                    } else {
                                        phi_1097_ = u32();
                                        phi_1100_ = u32();
                                        phi_1102_ = u32();
                                    }
                                    let _e869_ = phi_1097_;
                                    let _e871_ = phi_1100_;
                                    let _e873_ = phi_1102_;
                                    phi_1096_ = _e869_;
                                    phi_1099_ = _e871_;
                                    phi_1101_ = _e873_;
                                    if !(_e860_) {
                                        break;
                                    }
                                }
                                let _e2117_ = local_17_;
                                phi_1117_ = _e2117_;
                            }
                            let _e876_ = phi_1117_;
                            phi_1119_ = _e66_;
                            phi_1122_ = 0f;
                            phi_1124_ = 0f;
                            phi_1126_ = 0f;
                            loop {
                                let _e878_ = phi_1119_;
                                let _e880_ = phi_1122_;
                                let _e882_ = phi_1124_;
                                let _e884_ = phi_1126_;
                                local_5_ = _e884_;
                                local_6_ = _e882_;
                                local_7_ = _e880_;
                                let _e885_ = (_e878_ < _e69_);
                                if _e885_ {
                                    let _e887_ = ((_e876_ * _e69_) + _e878_);
                                    if (_e61_ == 0u) {
                                        phi_1146_ = 0u;
                                    } else {
                                        let _e889_ = (_e61_ - 1u);
                                        phi_1146_ = select(_e889_, _e887_, (_e887_ < _e889_));
                                    }
                                    let _e893_ = phi_1146_;
                                    if (_e893_ < _e61_) {
                                    } else {
                                        phi_3950_ = true;
                                        break;
                                    }
                                    let _e898_ = lbfgs_s_y.member[_e893_].x;
                                    let _e902_ = lbfgs_s_y.member[_e893_].y;
                                    if (_e63_ == 0u) {
                                        phi_1177_ = 0u;
                                    } else {
                                        let _e904_ = (_e63_ - 1u);
                                        phi_1177_ = select(_e904_, _e878_, (_e878_ < _e904_));
                                    }
                                    let _e908_ = phi_1177_;
                                    if (_e908_ < _e63_) {
                                    } else {
                                        phi_3950_ = true;
                                        break;
                                    }
                                    let _e912_ = search_dir.member[_e908_];
                                    if _e814_ {
                                        phi_1194_ = _e880_;
                                    } else {
                                        phi_1194_ = (_e880_ + (_e902_ * _e902_));
                                    }
                                    let _e920_ = phi_1194_;
                                    phi_1120_ = (_e878_ + 256u);
                                    phi_1123_ = _e920_;
                                    phi_1125_ = (_e882_ + (_e898_ * _e912_));
                                    phi_1127_ = (_e884_ + (_e898_ * _e902_));
                                } else {
                                    phi_1120_ = u32();
                                    phi_1123_ = f32();
                                    phi_1125_ = f32();
                                    phi_1127_ = f32();
                                }
                                let _e923_ = phi_1120_;
                                let _e925_ = phi_1123_;
                                let _e927_ = phi_1125_;
                                let _e929_ = phi_1127_;
                                phi_1119_ = _e923_;
                                phi_1122_ = _e925_;
                                phi_1124_ = _e927_;
                                phi_1126_ = _e929_;
                                phi_3950_ = _e808_;
                                if !(_e885_) {
                                    break;
                                }
                            }
                            let _e932_ = phi_3950_;
                            phi_4050_ = _e932_;
                            if _e932_ {
                                break;
                            }
                            let _e934_ = local_5_;
                            let _e935_ = bitcast<u32>(_e934_);
                            let _e936_ = (_e935_ & 2139095040u);
                            if (_e936_ == 2139095040u) {
                                phi_1206_ = true;
                            } else {
                                phi_1206_ = (_e936_ == 0u);
                            }
                            let _e940_ = phi_1206_;
                            let _e944_ = local_6_;
                            let _e945_ = bitcast<u32>(_e944_);
                            let _e946_ = (_e945_ & 2139095040u);
                            if (_e946_ == 2139095040u) {
                                phi_1222_ = true;
                            } else {
                                phi_1222_ = (_e946_ == 0u);
                            }
                            let _e950_ = phi_1222_;
                            let _e954_ = select(255u, _e66_, (_e66_ < 255u));
                            let _e955_ = (_e954_ < 256u);
                            if _e955_ {
                            } else {
                                phi_4050_ = true;
                                break;
                            }
                            scratch_q[_e954_] = bitcast<f32>(select(_e935_, 0u, _e940_));
                            if _e955_ {
                            } else {
                                phi_4050_ = true;
                                break;
                            }
                            dot_scratch[_e954_] = bitcast<f32>(select(_e945_, 0u, _e950_));
                            workgroupBarrier();
                            phi_1252_ = 128u;
                            loop {
                                let _e959_ = phi_1252_;
                                let _e960_ = (_e959_ > 0u);
                                if _e960_ {
                                    if (_e66_ < _e959_) {
                                        let _e962_ = (_e66_ + _e959_);
                                        let _e963_ = (_e962_ < 256u);
                                        if _e963_ {
                                        } else {
                                            phi_3984_ = true;
                                            break;
                                        }
                                        let _e965_ = scratch_q[_e962_];
                                        let _e966_ = (_e66_ < 256u);
                                        if _e966_ {
                                        } else {
                                            phi_3984_ = true;
                                            break;
                                        }
                                        let _e968_ = scratch_q[_e66_];
                                        scratch_q[_e66_] = (_e968_ + _e965_);
                                        if _e963_ {
                                        } else {
                                            phi_3984_ = true;
                                            break;
                                        }
                                        let _e971_ = dot_scratch[_e962_];
                                        if _e966_ {
                                        } else {
                                            phi_3984_ = true;
                                            break;
                                        }
                                        let _e973_ = dot_scratch[_e66_];
                                        dot_scratch[_e66_] = (_e973_ + _e971_);
                                    }
                                    workgroupBarrier();
                                    phi_1253_ = (_e959_ >> bitcast<u32>(1i));
                                } else {
                                    phi_1253_ = u32();
                                }
                                let _e978_ = phi_1253_;
                                phi_1252_ = _e978_;
                                phi_3984_ = _e932_;
                                if !(_e960_) {
                                    break;
                                }
                            }
                            let _e981_ = phi_3984_;
                            phi_4050_ = _e981_;
                            if _e981_ {
                                break;
                            }
                            let _e983_ = scratch_q[0];
                            let _e984_ = bitcast<u32>(_e983_);
                            let _e985_ = (_e984_ & 2139095040u);
                            if (_e985_ == 2139095040u) {
                                phi_1313_ = true;
                            } else {
                                phi_1313_ = (_e985_ == 0u);
                            }
                            let _e989_ = phi_1313_;
                            let _e991_ = bitcast<f32>(select(_e984_, 0u, _e989_));
                            let _e993_ = dot_scratch[0];
                            let _e994_ = bitcast<u32>(_e993_);
                            let _e995_ = (_e994_ & 2139095040u);
                            if (_e995_ == 2139095040u) {
                                phi_1331_ = true;
                            } else {
                                phi_1331_ = (_e995_ == 0u);
                            }
                            let _e999_ = phi_1331_;
                            workgroupBarrier();
                            if _e814_ {
                                phi_4019_ = _e981_;
                                phi_1428_ = 0f;
                            } else {
                                let _e1003_ = local_7_;
                                let _e1004_ = bitcast<u32>(_e1003_);
                                let _e1005_ = (_e1004_ & 2139095040u);
                                if (_e1005_ == 2139095040u) {
                                    phi_1350_ = true;
                                } else {
                                    phi_1350_ = (_e1005_ == 0u);
                                }
                                let _e1009_ = phi_1350_;
                                if _e955_ {
                                } else {
                                    phi_4050_ = true;
                                    break;
                                }
                                dot_scratch[_e954_] = bitcast<f32>(select(_e1004_, 0u, _e1009_));
                                workgroupBarrier();
                                phi_1374_ = 128u;
                                loop {
                                    let _e1013_ = phi_1374_;
                                    let _e1014_ = (_e1013_ > 0u);
                                    if _e1014_ {
                                        if (_e66_ < _e1013_) {
                                            let _e1016_ = (_e66_ < 256u);
                                            if _e1016_ {
                                            } else {
                                                phi_3994_ = true;
                                                break;
                                            }
                                            let _e1018_ = dot_scratch[_e66_];
                                            let _e1019_ = (_e66_ + _e1013_);
                                            if (_e1019_ < 256u) {
                                            } else {
                                                phi_3994_ = true;
                                                break;
                                            }
                                            let _e1022_ = dot_scratch[_e1019_];
                                            if _e1016_ {
                                            } else {
                                                phi_3994_ = true;
                                                break;
                                            }
                                            dot_scratch[_e66_] = (_e1018_ + _e1022_);
                                        }
                                        workgroupBarrier();
                                        phi_1375_ = (_e1013_ >> bitcast<u32>(1i));
                                    } else {
                                        phi_1375_ = u32();
                                    }
                                    let _e1027_ = phi_1375_;
                                    phi_1374_ = _e1027_;
                                    phi_3994_ = _e981_;
                                    if !(_e1014_) {
                                        break;
                                    }
                                }
                                let _e1030_ = phi_3994_;
                                phi_4050_ = _e1030_;
                                if _e1030_ {
                                    break;
                                }
                                let _e1031_ = dot_scratch[0];
                                let _e1032_ = bitcast<u32>(_e1031_);
                                let _e1033_ = (_e1032_ & 2139095040u);
                                if (_e1033_ == 2139095040u) {
                                    phi_1422_ = true;
                                } else {
                                    phi_1422_ = (_e1033_ == 0u);
                                }
                                let _e1037_ = phi_1422_;
                                workgroupBarrier();
                                phi_4019_ = _e1030_;
                                phi_1428_ = bitcast<f32>(select(_e1032_, 0u, _e1037_));
                            }
                            let _e1041_ = phi_4019_;
                            let _e1043_ = phi_1428_;
                            let _e1046_ = ((bitcast<u32>(_e991_) & 2139095040u) != 2139095040u);
                            if _e1046_ {
                                let _e1047_ = (_e991_ > 0f);
                                if _e1047_ {
                                    if _e1046_ {
                                        let _e1048_ = (_e991_ <= 0f);
                                        if _e1048_ {
                                            phi_1531_ = f32();
                                        } else {
                                            if (_e991_ < 0.00001f) {
                                                let _e1080_ = (select(0.000000000000000000000001f, _e991_, (_e991_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                if (_e1080_ != _e1080_) {
                                                    phi_3524_ = true;
                                                } else {
                                                    phi_3524_ = (0.000001f >= _e1080_);
                                                }
                                                let _e1084_ = phi_3524_;
                                                let _e1085_ = select(_e1080_, 0.000001f, _e1084_);
                                                let _e1088_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1085_)));
                                                let _e1092_ = (_e1088_ + (_e1088_ * (1f - (_e1085_ * _e1088_))));
                                                let _e1096_ = (_e1092_ + (_e1092_ * (1f - (_e1085_ * _e1092_))));
                                                let _e1102_ = bitcast<u32>(((_e1096_ + (_e1096_ * (1f - (_e1085_ * _e1096_)))) * 18446744000000000000f));
                                                let _e1103_ = (_e1102_ & 2139095040u);
                                                if (_e1103_ == 2139095040u) {
                                                    phi_1489_ = true;
                                                } else {
                                                    phi_1489_ = (_e1103_ == 0u);
                                                }
                                                let _e1107_ = phi_1489_;
                                                phi_1530_ = bitcast<f32>(select(_e1102_, 0u, _e1107_));
                                            } else {
                                                if (_e991_ != _e991_) {
                                                    phi_3464_ = true;
                                                } else {
                                                    phi_3464_ = (0.000001f >= _e991_);
                                                }
                                                let _e1053_ = phi_3464_;
                                                let _e1054_ = select(_e991_, 0.000001f, _e1053_);
                                                let _e1057_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1054_)));
                                                let _e1061_ = (_e1057_ + (_e1057_ * (1f - (_e1054_ * _e1057_))));
                                                let _e1065_ = (_e1061_ + (_e1061_ * (1f - (_e1054_ * _e1061_))));
                                                let _e1070_ = bitcast<u32>((_e1065_ + (_e1065_ * (1f - (_e1054_ * _e1065_)))));
                                                let _e1071_ = (_e1070_ & 2139095040u);
                                                if (_e1071_ == 2139095040u) {
                                                    phi_1524_ = true;
                                                } else {
                                                    phi_1524_ = (_e1071_ == 0u);
                                                }
                                                let _e1075_ = phi_1524_;
                                                phi_1530_ = bitcast<f32>(select(_e1070_, 0u, _e1075_));
                                            }
                                            let _e1111_ = phi_1530_;
                                            phi_1531_ = _e1111_;
                                        }
                                        let _e1113_ = phi_1531_;
                                        phi_1532_ = _e1113_;
                                        phi_1533_ = _e1048_;
                                    } else {
                                        phi_1532_ = f32();
                                        phi_1533_ = true;
                                    }
                                    let _e1115_ = phi_1532_;
                                    let _e1117_ = phi_1533_;
                                    let _e1118_ = select(_e1115_, 0f, _e1117_);
                                    let _e1120_ = bitcast<u32>((_e1118_ * bitcast<f32>(select(_e994_, 0u, _e999_))));
                                    let _e1121_ = (_e1120_ & 2139095040u);
                                    if (_e1121_ == 2139095040u) {
                                        phi_1549_ = true;
                                    } else {
                                        phi_1549_ = (_e1121_ == 0u);
                                    }
                                    let _e1125_ = phi_1549_;
                                    let _e1127_ = bitcast<f32>(select(_e1120_, 0u, _e1125_));
                                    if _e814_ {
                                        phi_1680_ = _e812_;
                                    } else {
                                        let _e1130_ = (_e1043_ + select(0.000000000001f, _e281_, (_e281_ > 0.000000000001f)));
                                        if ((bitcast<u32>(_e1130_) & 2139095040u) != 2139095040u) {
                                            let _e1134_ = (_e1130_ <= 0f);
                                            if _e1134_ {
                                                phi_1656_ = f32();
                                            } else {
                                                if (_e1130_ < 0.00001f) {
                                                    let _e1166_ = (select(0.000000000000000000000001f, _e1130_, (_e1130_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                    if (_e1166_ != _e1166_) {
                                                        phi_3644_ = true;
                                                    } else {
                                                        phi_3644_ = (0.000001f >= _e1166_);
                                                    }
                                                    let _e1170_ = phi_3644_;
                                                    let _e1171_ = select(_e1166_, 0.000001f, _e1170_);
                                                    let _e1174_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1171_)));
                                                    let _e1178_ = (_e1174_ + (_e1174_ * (1f - (_e1171_ * _e1174_))));
                                                    let _e1182_ = (_e1178_ + (_e1178_ * (1f - (_e1171_ * _e1178_))));
                                                    let _e1188_ = bitcast<u32>(((_e1182_ + (_e1182_ * (1f - (_e1171_ * _e1182_)))) * 18446744000000000000f));
                                                    let _e1189_ = (_e1188_ & 2139095040u);
                                                    if (_e1189_ == 2139095040u) {
                                                        phi_1614_ = true;
                                                    } else {
                                                        phi_1614_ = (_e1189_ == 0u);
                                                    }
                                                    let _e1193_ = phi_1614_;
                                                    phi_1655_ = bitcast<f32>(select(_e1188_, 0u, _e1193_));
                                                } else {
                                                    if (_e1130_ != _e1130_) {
                                                        phi_3584_ = true;
                                                    } else {
                                                        phi_3584_ = (0.000001f >= _e1130_);
                                                    }
                                                    let _e1139_ = phi_3584_;
                                                    let _e1140_ = select(_e1130_, 0.000001f, _e1139_);
                                                    let _e1143_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1140_)));
                                                    let _e1147_ = (_e1143_ + (_e1143_ * (1f - (_e1140_ * _e1143_))));
                                                    let _e1151_ = (_e1147_ + (_e1147_ * (1f - (_e1140_ * _e1147_))));
                                                    let _e1156_ = bitcast<u32>((_e1151_ + (_e1151_ * (1f - (_e1140_ * _e1151_)))));
                                                    let _e1157_ = (_e1156_ & 2139095040u);
                                                    if (_e1157_ == 2139095040u) {
                                                        phi_1649_ = true;
                                                    } else {
                                                        phi_1649_ = (_e1157_ == 0u);
                                                    }
                                                    let _e1161_ = phi_1649_;
                                                    phi_1655_ = bitcast<f32>(select(_e1156_, 0u, _e1161_));
                                                }
                                                let _e1197_ = phi_1655_;
                                                phi_1656_ = _e1197_;
                                            }
                                            let _e1199_ = phi_1656_;
                                            phi_1657_ = _e1199_;
                                            phi_1658_ = _e1134_;
                                        } else {
                                            phi_1657_ = f32();
                                            phi_1658_ = true;
                                        }
                                        let _e1201_ = phi_1657_;
                                        let _e1203_ = phi_1658_;
                                        let _e1206_ = bitcast<u32>((_e991_ * select(_e1201_, 0f, _e1203_)));
                                        let _e1207_ = (_e1206_ & 2139095040u);
                                        if (_e1207_ == 2139095040u) {
                                            phi_1674_ = true;
                                        } else {
                                            phi_1674_ = (_e1207_ == 0u);
                                        }
                                        let _e1211_ = phi_1674_;
                                        phi_1680_ = bitcast<f32>(select(_e1206_, 0u, _e1211_));
                                    }
                                    let _e1215_ = phi_1680_;
                                    phi_1683_ = _e66_;
                                    loop {
                                        let _e1218_ = phi_1683_;
                                        let _e1219_ = (_e1218_ < _e69_);
                                        if _e1219_ {
                                            let _e1221_ = ((_e876_ * _e69_) + _e1218_);
                                            if (_e61_ == 0u) {
                                                phi_1704_ = 0u;
                                            } else {
                                                let _e1223_ = (_e61_ - 1u);
                                                phi_1704_ = select(_e1223_, _e1221_, (_e1221_ < _e1223_));
                                            }
                                            let _e1227_ = phi_1704_;
                                            if (_e1227_ < _e61_) {
                                            } else {
                                                phi_4006_ = true;
                                                break;
                                            }
                                            let _e1232_ = lbfgs_s_y.member[_e1227_].y;
                                            if (_e63_ == 0u) {
                                                phi_1723_ = 0u;
                                            } else {
                                                let _e1234_ = (_e63_ - 1u);
                                                phi_1723_ = select(_e1234_, _e1218_, (_e1218_ < _e1234_));
                                            }
                                            let _e1238_ = phi_1723_;
                                            let _e1239_ = (_e1238_ < _e63_);
                                            if _e1239_ {
                                            } else {
                                                phi_4006_ = true;
                                                break;
                                            }
                                            let _e1242_ = search_dir.member[_e1238_];
                                            let _e1245_ = bitcast<u32>((_e1242_ - (_e1127_ * _e1232_)));
                                            let _e1246_ = (_e1245_ & 2139095040u);
                                            if (_e1246_ == 2139095040u) {
                                                phi_1743_ = true;
                                            } else {
                                                phi_1743_ = (_e1246_ == 0u);
                                            }
                                            let _e1250_ = phi_1743_;
                                            if _e1239_ {
                                            } else {
                                                phi_4006_ = true;
                                                break;
                                            }
                                            search_dir.member[_e1238_] = bitcast<f32>(select(_e1245_, 0u, _e1250_));
                                            phi_1684_ = (_e1218_ + 256u);
                                        } else {
                                            phi_1684_ = u32();
                                        }
                                        let _e1255_ = phi_1684_;
                                        phi_1683_ = _e1255_;
                                        phi_4006_ = _e1041_;
                                        if !(_e1219_) {
                                            break;
                                        }
                                    }
                                    let _e1258_ = phi_4006_;
                                    phi_4050_ = _e1258_;
                                    if _e1258_ {
                                        break;
                                    }
                                    phi_4055_ = _e1258_;
                                    phi_1756_ = _e1215_;
                                    phi_1757_ = _e1127_;
                                    phi_1758_ = _e1118_;
                                    phi_1759_ = select(true, _e814_, _e814_);
                                } else {
                                    phi_4055_ = _e1041_;
                                    phi_1756_ = f32();
                                    phi_1757_ = f32();
                                    phi_1758_ = f32();
                                    phi_1759_ = bool();
                                }
                                let _e1260_ = phi_4055_;
                                let _e1262_ = phi_1756_;
                                let _e1264_ = phi_1757_;
                                let _e1266_ = phi_1758_;
                                let _e1268_ = phi_1759_;
                                phi_4054_ = _e1260_;
                                phi_1761_ = _e1262_;
                                phi_1762_ = _e1264_;
                                phi_1763_ = _e1266_;
                                phi_1764_ = _e1268_;
                                phi_1765_ = select(true, false, _e1047_);
                            } else {
                                phi_4054_ = _e1041_;
                                phi_1761_ = f32();
                                phi_1762_ = f32();
                                phi_1763_ = f32();
                                phi_1764_ = bool();
                                phi_1765_ = true;
                            }
                            let _e1271_ = phi_4054_;
                            let _e1273_ = phi_1761_;
                            let _e1275_ = phi_1762_;
                            let _e1277_ = phi_1763_;
                            let _e1279_ = phi_1764_;
                            let _e1281_ = phi_1765_;
                            if (_e66_ == 0u) {
                                if (_e829_.member_1_ < 32u) {
                                } else {
                                    phi_4050_ = true;
                                    break;
                                }
                                lbfgs_scalars[_e829_.member_1_] = select(_e1275_, 0f, _e1281_);
                                let _e1289_ = (8u + _e829_.member_1_);
                                if (_e1289_ < 32u) {
                                } else {
                                    phi_4050_ = true;
                                    break;
                                }
                                lbfgs_scalars[_e1289_] = select(_e1277_, 0f, _e1281_);
                                let _e1292_ = (16u + _e829_.member_1_);
                                if (_e1292_ < 32u) {
                                } else {
                                    phi_4050_ = true;
                                    break;
                                }
                                lbfgs_scalars[_e1292_] = _e991_;
                            }
                            workgroupBarrier();
                            phi_4051_ = _e1271_;
                            phi_1040_ = select(_e1273_, _e812_, _e1281_);
                            phi_1042_ = select(_e1279_, _e814_, _e1281_);
                        } else {
                            phi_4051_ = _e808_;
                            phi_1040_ = f32();
                            phi_1042_ = bool();
                        }
                        let _e1296_ = phi_4051_;
                        let _e1298_ = phi_1040_;
                        let _e1300_ = phi_1042_;
                        phi_3959_ = _e1296_;
                        phi_1036_ = _e827_;
                        phi_1039_ = _e1298_;
                        phi_1041_ = _e1300_;
                        phi_4050_ = _e1296_;
                        if !(_e833_) {
                            break;
                        }
                    }
                    let _e1303_ = phi_4050_;
                    if _e1303_ {
                        break;
                    }
                    phi_1798_ = _e66_;
                    loop {
                        let _e1305_ = phi_1798_;
                        let _e1306_ = (_e1305_ < _e69_);
                        if _e1306_ {
                            if (_e63_ == 0u) {
                                phi_1817_ = 0u;
                            } else {
                                let _e1308_ = (_e63_ - 1u);
                                phi_1817_ = select(_e1308_, _e1305_, (_e1305_ < _e1308_));
                            }
                            let _e1312_ = phi_1817_;
                            let _e1313_ = (_e1312_ < _e63_);
                            if _e1313_ {
                            } else {
                                phi_4056_ = true;
                                break;
                            }
                            let _e1316_ = search_dir.member[_e1312_];
                            let _e1318_ = local_8_;
                            let _e1320_ = bitcast<u32>((_e1316_ * _e1318_));
                            let _e1321_ = (_e1320_ & 2139095040u);
                            if (_e1321_ == 2139095040u) {
                                phi_1836_ = true;
                            } else {
                                phi_1836_ = (_e1321_ == 0u);
                            }
                            let _e1325_ = phi_1836_;
                            if _e1313_ {
                            } else {
                                phi_4056_ = true;
                                break;
                            }
                            search_dir.member[_e1312_] = bitcast<f32>(select(_e1320_, 0u, _e1325_));
                            phi_1799_ = (_e1305_ + 256u);
                        } else {
                            phi_1799_ = u32();
                        }
                        let _e1330_ = phi_1799_;
                        phi_1798_ = _e1330_;
                        phi_4056_ = _e1303_;
                        if !(_e1306_) {
                            break;
                        }
                    }
                    let _e1333_ = phi_4056_;
                    if _e1333_ {
                        break;
                    }
                    let _e1334_ = (_e66_ == 0u);
                    if _e1334_ {
                        let _e1337_ = local_9_;
                        lbfgs_scalars[24] = _e1337_;
                        lbfgs_scalars[26] = f32(_e730_);
                    }
                    workgroupBarrier();
                    phi_4072_ = _e1333_;
                    phi_1858_ = _e806_;
                    loop {
                        let _e1341_ = phi_4072_;
                        let _e1343_ = phi_1858_;
                        if (_e1343_.member < _e1343_.member_1_) {
                            phi_1859_ = type_30_((_e1343_.member + 1u), _e1343_.member_1_);
                            phi_1882_ = type_30_(1u, _e1343_.member);
                        } else {
                            phi_1859_ = _e1343_;
                            phi_1882_ = type_30_(0u, u32());
                        }
                        let _e1356_ = phi_1859_;
                        let _e1358_ = phi_1882_;
                        let _e1362_ = (bitcast<i32>(_e1358_.member) != 0i);
                        if _e1362_ {
                            let _e1364_ = ((_e730_ - 1u) - _e1358_.member_1_);
                            let _e1366_ = ((_e305_ - 1u) - _e1364_);
                            if (_e267_ <= 1u) {
                                phi_1937_ = 0u;
                            } else {
                                phi_1901_ = 0u;
                                phi_1904_ = _e267_;
                                loop {
                                    let _e1369_ = phi_1901_;
                                    let _e1371_ = phi_1904_;
                                    local_18_ = _e1371_;
                                    local_10_ = _e1369_;
                                    let _e1374_ = (_e1371_ <= (_e1366_ >> bitcast<u32>(1i)));
                                    if _e1374_ {
                                        phi_1902_ = (_e1369_ + 1u);
                                        phi_1905_ = (_e1371_ << bitcast<u32>(1i));
                                    } else {
                                        phi_1902_ = u32();
                                        phi_1905_ = u32();
                                    }
                                    let _e1379_ = phi_1902_;
                                    let _e1381_ = phi_1905_;
                                    phi_1901_ = _e1379_;
                                    phi_1904_ = _e1381_;
                                    if !(_e1374_) {
                                        break;
                                    }
                                }
                                phi_1916_ = 0u;
                                let _e2205_ = local_18_;
                                phi_1919_ = _e2205_;
                                phi_1921_ = _e1366_;
                                loop {
                                    let _e1384_ = phi_1916_;
                                    let _e1386_ = phi_1919_;
                                    let _e1388_ = phi_1921_;
                                    local_19_ = _e1388_;
                                    let _e1390_ = local_10_;
                                    let _e1391_ = (_e1384_ <= _e1390_);
                                    if _e1391_ {
                                        if (_e1388_ >= _e1386_) {
                                            phi_1934_ = (_e1388_ - _e1386_);
                                        } else {
                                            phi_1934_ = _e1388_;
                                        }
                                        let _e1395_ = phi_1934_;
                                        phi_1917_ = (_e1384_ + 1u);
                                        phi_1920_ = (_e1386_ >> bitcast<u32>(1i));
                                        phi_1922_ = _e1395_;
                                    } else {
                                        phi_1917_ = u32();
                                        phi_1920_ = u32();
                                        phi_1922_ = u32();
                                    }
                                    let _e1400_ = phi_1917_;
                                    let _e1402_ = phi_1920_;
                                    let _e1404_ = phi_1922_;
                                    phi_1916_ = _e1400_;
                                    phi_1919_ = _e1402_;
                                    phi_1921_ = _e1404_;
                                    if !(_e1391_) {
                                        break;
                                    }
                                }
                                let _e2214_ = local_19_;
                                phi_1937_ = _e2214_;
                            }
                            let _e1407_ = phi_1937_;
                            phi_1939_ = _e66_;
                            phi_1942_ = 0f;
                            loop {
                                let _e1409_ = phi_1939_;
                                let _e1411_ = phi_1942_;
                                local_11_ = _e1411_;
                                let _e1412_ = (_e1409_ < _e69_);
                                if _e1412_ {
                                    let _e1414_ = ((_e1407_ * _e69_) + _e1409_);
                                    if (_e61_ == 0u) {
                                        phi_1962_ = 0u;
                                    } else {
                                        let _e1416_ = (_e61_ - 1u);
                                        phi_1962_ = select(_e1416_, _e1414_, (_e1414_ < _e1416_));
                                    }
                                    let _e1420_ = phi_1962_;
                                    if (_e1420_ < _e61_) {
                                    } else {
                                        phi_4063_ = true;
                                        break;
                                    }
                                    let _e1425_ = lbfgs_s_y.member[_e1420_].y;
                                    if (_e63_ == 0u) {
                                        phi_1981_ = 0u;
                                    } else {
                                        let _e1427_ = (_e63_ - 1u);
                                        phi_1981_ = select(_e1427_, _e1409_, (_e1409_ < _e1427_));
                                    }
                                    let _e1431_ = phi_1981_;
                                    if (_e1431_ < _e63_) {
                                    } else {
                                        phi_4063_ = true;
                                        break;
                                    }
                                    let _e1435_ = search_dir.member[_e1431_];
                                    phi_1940_ = (_e1409_ + 256u);
                                    phi_1943_ = (_e1411_ + (_e1425_ * _e1435_));
                                } else {
                                    phi_1940_ = u32();
                                    phi_1943_ = f32();
                                }
                                let _e1440_ = phi_1940_;
                                let _e1442_ = phi_1943_;
                                phi_1939_ = _e1440_;
                                phi_1942_ = _e1442_;
                                phi_4063_ = _e1341_;
                                if !(_e1412_) {
                                    break;
                                }
                            }
                            let _e1445_ = phi_4063_;
                            phi_4100_ = _e1445_;
                            if _e1445_ {
                                break;
                            }
                            let _e1447_ = local_11_;
                            let _e1448_ = bitcast<u32>(_e1447_);
                            let _e1449_ = (_e1448_ & 2139095040u);
                            if (_e1449_ == 2139095040u) {
                                phi_2002_ = true;
                            } else {
                                phi_2002_ = (_e1449_ == 0u);
                            }
                            let _e1453_ = phi_2002_;
                            let _e1457_ = select(255u, _e66_, (_e66_ < 255u));
                            if (_e1457_ < 256u) {
                            } else {
                                phi_4100_ = true;
                                break;
                            }
                            dot_scratch[_e1457_] = bitcast<f32>(select(_e1448_, 0u, _e1453_));
                            workgroupBarrier();
                            phi_2026_ = 128u;
                            loop {
                                let _e1461_ = phi_2026_;
                                let _e1462_ = (_e1461_ > 0u);
                                if _e1462_ {
                                    if (_e66_ < _e1461_) {
                                        let _e1464_ = (_e66_ < 256u);
                                        if _e1464_ {
                                        } else {
                                            phi_4079_ = true;
                                            break;
                                        }
                                        let _e1466_ = dot_scratch[_e66_];
                                        let _e1467_ = (_e66_ + _e1461_);
                                        if (_e1467_ < 256u) {
                                        } else {
                                            phi_4079_ = true;
                                            break;
                                        }
                                        let _e1470_ = dot_scratch[_e1467_];
                                        if _e1464_ {
                                        } else {
                                            phi_4079_ = true;
                                            break;
                                        }
                                        dot_scratch[_e66_] = (_e1466_ + _e1470_);
                                    }
                                    workgroupBarrier();
                                    phi_2027_ = (_e1461_ >> bitcast<u32>(1i));
                                } else {
                                    phi_2027_ = u32();
                                }
                                let _e1475_ = phi_2027_;
                                phi_2026_ = _e1475_;
                                phi_4079_ = _e1445_;
                                if !(_e1462_) {
                                    break;
                                }
                            }
                            let _e1478_ = phi_4079_;
                            phi_4100_ = _e1478_;
                            if _e1478_ {
                                break;
                            }
                            let _e1480_ = dot_scratch[0];
                            let _e1481_ = bitcast<u32>(_e1480_);
                            let _e1482_ = (_e1481_ & 2139095040u);
                            if (_e1482_ == 2139095040u) {
                                phi_2074_ = true;
                            } else {
                                phi_2074_ = (_e1482_ == 0u);
                            }
                            let _e1486_ = phi_2074_;
                            workgroupBarrier();
                            if (_e1364_ < 32u) {
                            } else {
                                phi_4100_ = true;
                                break;
                            }
                            let _e1491_ = lbfgs_scalars[_e1364_];
                            let _e1492_ = (8u + _e1364_);
                            if (_e1492_ < 32u) {
                            } else {
                                phi_4100_ = true;
                                break;
                            }
                            let _e1495_ = lbfgs_scalars[_e1492_];
                            let _e1497_ = bitcast<u32>((_e1495_ * bitcast<f32>(select(_e1481_, 0u, _e1486_))));
                            let _e1498_ = (_e1497_ & 2139095040u);
                            if (_e1498_ == 2139095040u) {
                                phi_2106_ = true;
                            } else {
                                phi_2106_ = (_e1498_ == 0u);
                            }
                            let _e1502_ = phi_2106_;
                            phi_2114_ = _e66_;
                            loop {
                                let _e1507_ = phi_2114_;
                                let _e1508_ = (_e1507_ < _e69_);
                                if _e1508_ {
                                    let _e1510_ = ((_e1407_ * _e69_) + _e1507_);
                                    if (_e61_ == 0u) {
                                        phi_2135_ = 0u;
                                    } else {
                                        let _e1512_ = (_e61_ - 1u);
                                        phi_2135_ = select(_e1512_, _e1510_, (_e1510_ < _e1512_));
                                    }
                                    let _e1516_ = phi_2135_;
                                    if (_e1516_ < _e61_) {
                                    } else {
                                        phi_4087_ = true;
                                        break;
                                    }
                                    let _e1521_ = lbfgs_s_y.member[_e1516_].x;
                                    if (_e63_ == 0u) {
                                        phi_2154_ = 0u;
                                    } else {
                                        let _e1523_ = (_e63_ - 1u);
                                        phi_2154_ = select(_e1523_, _e1507_, (_e1507_ < _e1523_));
                                    }
                                    let _e1527_ = phi_2154_;
                                    let _e1528_ = (_e1527_ < _e63_);
                                    if _e1528_ {
                                    } else {
                                        phi_4087_ = true;
                                        break;
                                    }
                                    let _e1531_ = search_dir.member[_e1527_];
                                    let _e1534_ = bitcast<u32>((_e1531_ + (_e1521_ * (_e1491_ - bitcast<f32>(select(_e1497_, 0u, _e1502_))))));
                                    let _e1535_ = (_e1534_ & 2139095040u);
                                    if (_e1535_ == 2139095040u) {
                                        phi_2174_ = true;
                                    } else {
                                        phi_2174_ = (_e1535_ == 0u);
                                    }
                                    let _e1539_ = phi_2174_;
                                    if _e1528_ {
                                    } else {
                                        phi_4087_ = true;
                                        break;
                                    }
                                    search_dir.member[_e1527_] = bitcast<f32>(select(_e1534_, 0u, _e1539_));
                                    phi_2115_ = (_e1507_ + 256u);
                                } else {
                                    phi_2115_ = u32();
                                }
                                let _e1544_ = phi_2115_;
                                phi_2114_ = _e1544_;
                                phi_4087_ = _e1478_;
                                if !(_e1508_) {
                                    break;
                                }
                            }
                            let _e1547_ = phi_4087_;
                            phi_4100_ = _e1547_;
                            if _e1547_ {
                                break;
                            }
                            workgroupBarrier();
                            phi_4101_ = _e1547_;
                        } else {
                            phi_4101_ = _e1341_;
                        }
                        let _e1549_ = phi_4101_;
                        phi_4072_ = _e1549_;
                        phi_1858_ = _e1356_;
                        phi_4100_ = _e1549_;
                        if !(_e1362_) {
                            break;
                        }
                    }
                    let _e1552_ = phi_4100_;
                    if _e1552_ {
                        break;
                    }
                    phi_2188_ = _e66_;
                    phi_2191_ = 0f;
                    phi_2193_ = 0f;
                    loop {
                        let _e1554_ = phi_2188_;
                        let _e1556_ = phi_2191_;
                        let _e1558_ = phi_2193_;
                        local_12_ = _e1558_;
                        local_13_ = _e1556_;
                        let _e1559_ = (_e1554_ < _e69_);
                        if _e1559_ {
                            if (_e57_ == 0u) {
                                phi_2211_ = 0u;
                            } else {
                                let _e1561_ = (_e57_ - 1u);
                                phi_2211_ = select(_e1561_, _e1554_, (_e1554_ < _e1561_));
                            }
                            let _e1565_ = phi_2211_;
                            if (_e1565_ < _e57_) {
                            } else {
                                phi_4102_ = true;
                                break;
                            }
                            let _e1569_ = grad.member[_e1565_];
                            let _e1570_ = bitcast<u32>(_e1569_);
                            let _e1571_ = (_e1570_ & 2139095040u);
                            if (_e1571_ == 2139095040u) {
                                phi_2229_ = true;
                            } else {
                                phi_2229_ = (_e1571_ == 0u);
                            }
                            let _e1575_ = phi_2229_;
                            if (_e63_ == 0u) {
                                phi_2245_ = 0u;
                            } else {
                                let _e1579_ = (_e63_ - 1u);
                                phi_2245_ = select(_e1579_, _e1554_, (_e1554_ < _e1579_));
                            }
                            let _e1583_ = phi_2245_;
                            if (_e1583_ < _e63_) {
                            } else {
                                phi_4102_ = true;
                                break;
                            }
                            let _e1587_ = search_dir.member[_e1583_];
                            if ((bitcast<u32>(_e1587_) & 2139095040u) != 2139095040u) {
                                phi_2260_ = _e1556_;
                            } else {
                                phi_2260_ = (_e1556_ + 1f);
                            }
                            let _e1593_ = phi_2260_;
                            let _e1595_ = bitcast<u32>(-(_e1587_));
                            let _e1596_ = (_e1595_ & 2139095040u);
                            if (_e1596_ == 2139095040u) {
                                phi_2272_ = true;
                            } else {
                                phi_2272_ = (_e1596_ == 0u);
                            }
                            let _e1600_ = phi_2272_;
                            phi_2189_ = (_e1554_ + 256u);
                            phi_2192_ = _e1593_;
                            phi_2194_ = (_e1558_ + (bitcast<f32>(select(_e1570_, 0u, _e1575_)) * bitcast<f32>(select(_e1595_, 0u, _e1600_))));
                        } else {
                            phi_2189_ = u32();
                            phi_2192_ = f32();
                            phi_2194_ = f32();
                        }
                        let _e1607_ = phi_2189_;
                        let _e1609_ = phi_2192_;
                        let _e1611_ = phi_2194_;
                        phi_2188_ = _e1607_;
                        phi_2191_ = _e1609_;
                        phi_2193_ = _e1611_;
                        phi_4102_ = _e1552_;
                        if !(_e1559_) {
                            break;
                        }
                    }
                    let _e1614_ = phi_4102_;
                    if _e1614_ {
                        break;
                    }
                    let _e1616_ = local_12_;
                    let _e1617_ = bitcast<u32>(_e1616_);
                    let _e1618_ = (_e1617_ & 2139095040u);
                    if (_e1618_ == 2139095040u) {
                        phi_2291_ = true;
                    } else {
                        phi_2291_ = (_e1618_ == 0u);
                    }
                    let _e1622_ = phi_2291_;
                    let _e1626_ = local_13_;
                    let _e1627_ = bitcast<u32>(_e1626_);
                    let _e1628_ = (_e1627_ & 2139095040u);
                    if (_e1628_ == 2139095040u) {
                        phi_2307_ = true;
                    } else {
                        phi_2307_ = (_e1628_ == 0u);
                    }
                    let _e1632_ = phi_2307_;
                    let _e1636_ = select(255u, _e66_, (_e66_ < 255u));
                    let _e1637_ = (_e1636_ < 256u);
                    if _e1637_ {
                    } else {
                        break;
                    }
                    scratch_q[_e1636_] = bitcast<f32>(select(_e1617_, 0u, _e1622_));
                    if _e1637_ {
                    } else {
                        break;
                    }
                    dot_scratch[_e1636_] = bitcast<f32>(select(_e1627_, 0u, _e1632_));
                    workgroupBarrier();
                    phi_2337_ = 128u;
                    loop {
                        let _e1641_ = phi_2337_;
                        let _e1642_ = (_e1641_ > 0u);
                        if _e1642_ {
                            if (_e66_ < _e1641_) {
                                let _e1644_ = (_e66_ + _e1641_);
                                let _e1645_ = (_e1644_ < 256u);
                                if _e1645_ {
                                } else {
                                    phi_4114_ = true;
                                    break;
                                }
                                let _e1647_ = scratch_q[_e1644_];
                                let _e1648_ = (_e66_ < 256u);
                                if _e1648_ {
                                } else {
                                    phi_4114_ = true;
                                    break;
                                }
                                let _e1650_ = scratch_q[_e66_];
                                scratch_q[_e66_] = (_e1650_ + _e1647_);
                                if _e1645_ {
                                } else {
                                    phi_4114_ = true;
                                    break;
                                }
                                let _e1653_ = dot_scratch[_e1644_];
                                if _e1648_ {
                                } else {
                                    phi_4114_ = true;
                                    break;
                                }
                                let _e1655_ = dot_scratch[_e66_];
                                dot_scratch[_e66_] = (_e1655_ + _e1653_);
                            }
                            workgroupBarrier();
                            phi_2338_ = (_e1641_ >> bitcast<u32>(1i));
                        } else {
                            phi_2338_ = u32();
                        }
                        let _e1660_ = phi_2338_;
                        phi_2337_ = _e1660_;
                        phi_4114_ = _e1614_;
                        if !(_e1642_) {
                            break;
                        }
                    }
                    let _e1663_ = phi_4114_;
                    if _e1663_ {
                        break;
                    }
                    let _e1665_ = scratch_q[0];
                    let _e1666_ = bitcast<u32>(_e1665_);
                    let _e1667_ = (_e1666_ & 2139095040u);
                    if (_e1667_ == 2139095040u) {
                        phi_2398_ = true;
                    } else {
                        phi_2398_ = (_e1667_ == 0u);
                    }
                    let _e1671_ = phi_2398_;
                    let _e1673_ = bitcast<f32>(select(_e1666_, 0u, _e1671_));
                    let _e1675_ = dot_scratch[0];
                    let _e1676_ = bitcast<u32>(_e1675_);
                    let _e1677_ = (_e1676_ & 2139095040u);
                    if (_e1677_ == 2139095040u) {
                        phi_2416_ = true;
                    } else {
                        phi_2416_ = (_e1677_ == 0u);
                    }
                    let _e1681_ = phi_2416_;
                    workgroupBarrier();
                    if _e1334_ {
                        lbfgs_scalars[25] = _e1673_;
                    }
                    workgroupBarrier();
                    if ((bitcast<u32>(_e1673_) & 2139095040u) != 2139095040u) {
                        let _e1688_ = (_e1673_ >= 0f);
                        if _e1688_ {
                            phi_2438_ = bool();
                        } else {
                            phi_2438_ = (bitcast<f32>(select(_e1676_, 0u, _e1681_)) > 0f);
                        }
                        let _e1691_ = phi_2438_;
                        phi_2439_ = _e1691_;
                        phi_2440_ = _e1688_;
                    } else {
                        phi_2439_ = bool();
                        phi_2440_ = true;
                    }
                    let _e1693_ = phi_2439_;
                    let _e1695_ = phi_2440_;
                    if _e734_ {
                        let _e1697_ = (_e731_ <= 0f);
                        if _e1697_ {
                            phi_2538_ = f32();
                        } else {
                            if (_e731_ < 0.00001f) {
                                let _e1729_ = (select(0.000000000000000000000001f, _e731_, (_e731_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                if (_e1729_ != _e1729_) {
                                    phi_3764_ = true;
                                } else {
                                    phi_3764_ = (0.000001f >= _e1729_);
                                }
                                let _e1733_ = phi_3764_;
                                let _e1734_ = select(_e1729_, 0.000001f, _e1733_);
                                let _e1737_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1734_)));
                                let _e1741_ = (_e1737_ + (_e1737_ * (1f - (_e1734_ * _e1737_))));
                                let _e1745_ = (_e1741_ + (_e1741_ * (1f - (_e1734_ * _e1741_))));
                                let _e1751_ = bitcast<u32>(((_e1745_ + (_e1745_ * (1f - (_e1734_ * _e1745_)))) * 18446744000000000000f));
                                let _e1752_ = (_e1751_ & 2139095040u);
                                if (_e1752_ == 2139095040u) {
                                    phi_2496_ = true;
                                } else {
                                    phi_2496_ = (_e1752_ == 0u);
                                }
                                let _e1756_ = phi_2496_;
                                phi_2537_ = bitcast<f32>(select(_e1751_, 0u, _e1756_));
                            } else {
                                if (_e731_ != _e731_) {
                                    phi_3704_ = true;
                                } else {
                                    phi_3704_ = (0.000001f >= _e731_);
                                }
                                let _e1702_ = phi_3704_;
                                let _e1703_ = select(_e731_, 0.000001f, _e1702_);
                                let _e1706_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1703_)));
                                let _e1710_ = (_e1706_ + (_e1706_ * (1f - (_e1703_ * _e1706_))));
                                let _e1714_ = (_e1710_ + (_e1710_ * (1f - (_e1703_ * _e1710_))));
                                let _e1719_ = bitcast<u32>((_e1714_ + (_e1714_ * (1f - (_e1703_ * _e1714_)))));
                                let _e1720_ = (_e1719_ & 2139095040u);
                                if (_e1720_ == 2139095040u) {
                                    phi_2531_ = true;
                                } else {
                                    phi_2531_ = (_e1720_ == 0u);
                                }
                                let _e1724_ = phi_2531_;
                                phi_2537_ = bitcast<f32>(select(_e1719_, 0u, _e1724_));
                            }
                            let _e1760_ = phi_2537_;
                            phi_2538_ = _e1760_;
                        }
                        let _e1762_ = phi_2538_;
                        phi_2539_ = _e1762_;
                        phi_2540_ = _e1697_;
                    } else {
                        phi_2539_ = f32();
                        phi_2540_ = true;
                    }
                    let _e1764_ = phi_2539_;
                    let _e1766_ = phi_2540_;
                    phi_2551_ = _e66_;
                    loop {
                        let _e1771_ = phi_2551_;
                        let _e1772_ = (_e1771_ < _e69_);
                        if _e1772_ {
                            let _e1773_ = (_e55_ == 0u);
                            if _e1773_ {
                                phi_2570_ = 0u;
                            } else {
                                let _e1774_ = (_e55_ - 1u);
                                phi_2570_ = select(_e1774_, _e1771_, (_e1771_ < _e1774_));
                            }
                            let _e1778_ = phi_2570_;
                            if (_e1778_ < _e55_) {
                            } else {
                                phi_4124_ = true;
                                break;
                            }
                            let _e1782_ = theta.member[_e1778_];
                            let _e1783_ = bitcast<u32>(_e1782_);
                            let _e1784_ = (_e1783_ & 2139095040u);
                            if (_e1784_ == 2139095040u) {
                                phi_2588_ = true;
                            } else {
                                phi_2588_ = (_e1784_ == 0u);
                            }
                            let _e1788_ = phi_2588_;
                            let _e1790_ = bitcast<f32>(select(_e1783_, 0u, _e1788_));
                            if (_e57_ == 0u) {
                                phi_2604_ = 0u;
                            } else {
                                let _e1792_ = (_e57_ - 1u);
                                phi_2604_ = select(_e1792_, _e1771_, (_e1771_ < _e1792_));
                            }
                            let _e1796_ = phi_2604_;
                            if (_e1796_ < _e57_) {
                            } else {
                                phi_4124_ = true;
                                break;
                            }
                            let _e1800_ = grad.member[_e1796_];
                            let _e1801_ = bitcast<u32>(_e1800_);
                            let _e1802_ = (_e1801_ & 2139095040u);
                            if (_e1802_ == 2139095040u) {
                                phi_2622_ = true;
                            } else {
                                phi_2622_ = (_e1802_ == 0u);
                            }
                            let _e1806_ = phi_2622_;
                            let _e1808_ = bitcast<f32>(select(_e1801_, 0u, _e1806_));
                            if (_e59_ == 0u) {
                                phi_2638_ = 0u;
                            } else {
                                let _e1810_ = (_e59_ - 1u);
                                phi_2638_ = select(_e1810_, _e1771_, (_e1771_ < _e1810_));
                            }
                            let _e1814_ = phi_2638_;
                            if (_e1814_ < _e59_) {
                            } else {
                                phi_4124_ = true;
                                break;
                            }
                            let _e1819_ = param_bounds.member[_e1814_].x;
                            let _e1823_ = param_bounds.member[_e1814_].y;
                            let _e1824_ = (_e63_ == 0u);
                            if _e1824_ {
                                phi_2659_ = 0u;
                            } else {
                                let _e1825_ = (_e63_ - 1u);
                                phi_2659_ = select(_e1825_, _e1771_, (_e1771_ < _e1825_));
                            }
                            let _e1829_ = phi_2659_;
                            if (_e1829_ < _e63_) {
                            } else {
                                phi_4124_ = true;
                                break;
                            }
                            let _e1833_ = search_dir.member[_e1829_];
                            if select(_e1693_, true, _e1695_) {
                                let _e1845_ = bitcast<u32>((-(_e1808_) * select(_e1764_, 0f, _e1766_)));
                                let _e1846_ = (_e1845_ & 2139095040u);
                                if (_e1846_ == 2139095040u) {
                                    phi_2682_ = true;
                                } else {
                                    phi_2682_ = (_e1846_ == 0u);
                                }
                                let _e1850_ = phi_2682_;
                                phi_2705_ = bitcast<f32>(select(_e1845_, 0u, _e1850_));
                            } else {
                                let _e1835_ = bitcast<u32>(-(_e1833_));
                                let _e1836_ = (_e1835_ & 2139095040u);
                                if (_e1836_ == 2139095040u) {
                                    phi_2699_ = true;
                                } else {
                                    phi_2699_ = (_e1836_ == 0u);
                                }
                                let _e1840_ = phi_2699_;
                                phi_2705_ = bitcast<f32>(select(_e1835_, 0u, _e1840_));
                            }
                            let _e1854_ = phi_2705_;
                            let _e1856_ = ((_e267_ * _e69_) + _e1771_);
                            if (_e61_ == 0u) {
                                phi_2719_ = 0u;
                            } else {
                                let _e1859_ = (_e61_ - 1u);
                                phi_2719_ = select(_e1859_, _e1856_, (_e1856_ < _e1859_));
                            }
                            let _e1863_ = phi_2719_;
                            if (_e1863_ < _e61_) {
                            } else {
                                phi_4124_ = true;
                                break;
                            }
                            lbfgs_s_y.member[_e1863_] = vec4<f32>(_e1790_, _e1808_, select(0f, 1f, (_e730_ > 0u)), 0f);
                            if _e1824_ {
                                phi_2736_ = 0u;
                            } else {
                                let _e1867_ = (_e63_ - 1u);
                                phi_2736_ = select(_e1867_, _e1771_, (_e1771_ < _e1867_));
                            }
                            let _e1871_ = phi_2736_;
                            if (_e1871_ < _e63_) {
                            } else {
                                phi_4124_ = true;
                                break;
                            }
                            search_dir.member[_e1871_] = _e1854_;
                            let _e1876_ = (_e1790_ + (select(0f, _e298_, ((bitcast<u32>(_e298_) & 2139095040u) != 2139095040u)) * _e1854_));
                            let _e1880_ = select(-1000000f, _e1819_, ((bitcast<u32>(_e1819_) & 2139095040u) != 2139095040u));
                            let _e1884_ = select(1000000f, _e1823_, ((bitcast<u32>(_e1823_) & 2139095040u) != 2139095040u));
                            let _e1885_ = (_e1880_ <= _e1884_);
                            let _e1886_ = select(_e1884_, _e1880_, _e1885_);
                            let _e1887_ = select(_e1880_, _e1884_, _e1885_);
                            let _e1888_ = (_e1886_ + _e295_);
                            let _e1889_ = (_e1887_ - _e295_);
                            let _e1890_ = (_e1888_ <= _e1889_);
                            if _e1890_ {
                                let _e1901_ = select(_e1889_, _e1888_, _e1890_);
                                let _e1902_ = select(_e1888_, _e1889_, _e1890_);
                                let _e1903_ = bitcast<u32>(_e1876_);
                                let _e1904_ = (_e1903_ & 2139095040u);
                                let _e1905_ = (_e1904_ == 2139095040u);
                                if _e1905_ {
                                    if ((_e1903_ & 2147483647u) > 2139095040u) {
                                        phi_2819_ = f32();
                                        phi_2820_ = true;
                                    } else {
                                        let _e1936_ = ((_e1903_ & 2147483648u) == 0u);
                                        if _e1936_ {
                                            let _e1937_ = bitcast<u32>(_e1902_);
                                            let _e1938_ = (_e1937_ & 2139095040u);
                                            if (_e1938_ == 2139095040u) {
                                                phi_2811_ = true;
                                            } else {
                                                phi_2811_ = (_e1938_ == 0u);
                                            }
                                            let _e1942_ = phi_2811_;
                                            phi_2817_ = bitcast<f32>(select(_e1937_, 0u, _e1942_));
                                        } else {
                                            phi_2817_ = f32();
                                        }
                                        let _e1946_ = phi_2817_;
                                        phi_2819_ = _e1946_;
                                        phi_2820_ = select(true, false, _e1936_);
                                    }
                                    let _e1949_ = phi_2819_;
                                    let _e1951_ = phi_2820_;
                                    if _e1951_ {
                                        let _e1952_ = bitcast<u32>(_e1901_);
                                        let _e1953_ = (_e1952_ & 2139095040u);
                                        if (_e1953_ == 2139095040u) {
                                            phi_2834_ = true;
                                        } else {
                                            phi_2834_ = (_e1953_ == 0u);
                                        }
                                        let _e1957_ = phi_2834_;
                                        phi_2840_ = bitcast<f32>(select(_e1952_, 0u, _e1957_));
                                    } else {
                                        phi_2840_ = _e1949_;
                                    }
                                    let _e1961_ = phi_2840_;
                                    phi_2899_ = _e1961_;
                                } else {
                                    if (_e1876_ < _e1901_) {
                                        let _e1923_ = bitcast<u32>(_e1901_);
                                        let _e1924_ = (_e1923_ & 2139095040u);
                                        if (_e1924_ == 2139095040u) {
                                            phi_2855_ = true;
                                        } else {
                                            phi_2855_ = (_e1924_ == 0u);
                                        }
                                        let _e1928_ = phi_2855_;
                                        phi_2898_ = bitcast<f32>(select(_e1923_, 0u, _e1928_));
                                    } else {
                                        if (_e1876_ > _e1902_) {
                                            let _e1913_ = bitcast<u32>(_e1902_);
                                            let _e1914_ = (_e1913_ & 2139095040u);
                                            if (_e1914_ == 2139095040u) {
                                                phi_2875_ = true;
                                            } else {
                                                phi_2875_ = (_e1914_ == 0u);
                                            }
                                            let _e1918_ = phi_2875_;
                                            phi_2897_ = bitcast<f32>(select(_e1913_, 0u, _e1918_));
                                        } else {
                                            if _e1905_ {
                                                phi_2891_ = true;
                                            } else {
                                                phi_2891_ = (_e1904_ == 0u);
                                            }
                                            let _e1910_ = phi_2891_;
                                            phi_2897_ = bitcast<f32>(select(_e1903_, 0u, _e1910_));
                                        }
                                        let _e1922_ = phi_2897_;
                                        phi_2898_ = _e1922_;
                                    }
                                    let _e1932_ = phi_2898_;
                                    phi_2899_ = _e1932_;
                                }
                                let _e1963_ = phi_2899_;
                                phi_2918_ = _e1963_;
                            } else {
                                let _e1893_ = bitcast<u32>((0.5f * (_e1886_ + _e1887_)));
                                let _e1894_ = (_e1893_ & 2139095040u);
                                if (_e1894_ == 2139095040u) {
                                    phi_2912_ = true;
                                } else {
                                    phi_2912_ = (_e1894_ == 0u);
                                }
                                let _e1898_ = phi_2912_;
                                phi_2918_ = bitcast<f32>(select(_e1893_, 0u, _e1898_));
                            }
                            let _e1965_ = phi_2918_;
                            if _e1773_ {
                                phi_2929_ = 0u;
                            } else {
                                let _e1966_ = (_e55_ - 1u);
                                phi_2929_ = select(_e1966_, _e1771_, (_e1771_ < _e1966_));
                            }
                            let _e1970_ = phi_2929_;
                            if (_e1970_ < _e55_) {
                            } else {
                                phi_4124_ = true;
                                break;
                            }
                            theta.member[_e1970_] = _e1965_;
                            phi_2552_ = (_e1771_ + 256u);
                        } else {
                            phi_2552_ = u32();
                        }
                        let _e1976_ = phi_2552_;
                        phi_2551_ = _e1976_;
                        phi_4124_ = _e1663_;
                        if !(_e1772_) {
                            break;
                        }
                    }
                    let _e1979_ = phi_4124_;
                    if _e1979_ {
                        break;
                    }
                } else {
                    let _e76_ = uniforms.member.member_1_;
                    let _e81_ = uniforms.member.member_6_;
                    if ((bitcast<u32>(_e81_) & 2139095040u) != 2139095040u) {
                        let _e85_ = (_e81_ >= 0f);
                        phi_2962_ = select(f32(), _e81_, _e85_);
                        phi_2963_ = select(true, false, _e85_);
                    } else {
                        phi_2962_ = f32();
                        phi_2963_ = true;
                    }
                    let _e89_ = phi_2962_;
                    let _e91_ = phi_2963_;
                    let _e92_ = select(_e89_, 0f, _e91_);
                    let _e95_ = uniforms.member.member_4_;
                    phi_2980_ = _e66_;
                    loop {
                        let _e101_ = phi_2980_;
                        let _e102_ = (_e101_ < _e69_);
                        if _e102_ {
                            let _e104_ = ((select(_e76_, 8u, (8u < _e76_)) * _e69_) + _e101_);
                            if (_e61_ == 0u) {
                                phi_3001_ = 0u;
                            } else {
                                let _e106_ = (_e61_ - 1u);
                                phi_3001_ = select(_e106_, _e104_, (_e104_ < _e106_));
                            }
                            let _e110_ = phi_3001_;
                            if (_e110_ < _e61_) {
                            } else {
                                phi_3814_ = true;
                                break;
                            }
                            let _e115_ = lbfgs_s_y.member[_e110_].x;
                            let _e116_ = bitcast<u32>(_e115_);
                            let _e117_ = (_e116_ & 2139095040u);
                            if (_e117_ == 2139095040u) {
                                phi_3020_ = true;
                            } else {
                                phi_3020_ = (_e117_ == 0u);
                            }
                            let _e121_ = phi_3020_;
                            if (_e63_ == 0u) {
                                phi_3036_ = 0u;
                            } else {
                                let _e125_ = (_e63_ - 1u);
                                phi_3036_ = select(_e125_, _e101_, (_e101_ < _e125_));
                            }
                            let _e129_ = phi_3036_;
                            if (_e129_ < _e63_) {
                            } else {
                                phi_3814_ = true;
                                break;
                            }
                            let _e133_ = search_dir.member[_e129_];
                            let _e134_ = bitcast<u32>(_e133_);
                            let _e135_ = (_e134_ & 2139095040u);
                            if (_e135_ == 2139095040u) {
                                phi_3054_ = true;
                            } else {
                                phi_3054_ = (_e135_ == 0u);
                            }
                            let _e139_ = phi_3054_;
                            if (_e59_ == 0u) {
                                phi_3070_ = 0u;
                            } else {
                                let _e143_ = (_e59_ - 1u);
                                phi_3070_ = select(_e143_, _e101_, (_e101_ < _e143_));
                            }
                            let _e147_ = phi_3070_;
                            if (_e147_ < _e59_) {
                            } else {
                                phi_3814_ = true;
                                break;
                            }
                            let _e152_ = param_bounds.member[_e147_].x;
                            let _e156_ = param_bounds.member[_e147_].y;
                            let _e158_ = (bitcast<f32>(select(_e116_, 0u, _e121_)) + (select(0f, _e95_, ((bitcast<u32>(_e95_) & 2139095040u) != 2139095040u)) * bitcast<f32>(select(_e134_, 0u, _e139_))));
                            let _e162_ = select(-1000000f, _e152_, ((bitcast<u32>(_e152_) & 2139095040u) != 2139095040u));
                            let _e166_ = select(1000000f, _e156_, ((bitcast<u32>(_e156_) & 2139095040u) != 2139095040u));
                            let _e167_ = (_e162_ <= _e166_);
                            let _e168_ = select(_e166_, _e162_, _e167_);
                            let _e169_ = select(_e162_, _e166_, _e167_);
                            let _e170_ = (_e168_ + _e92_);
                            let _e171_ = (_e169_ - _e92_);
                            let _e172_ = (_e170_ <= _e171_);
                            if _e172_ {
                                let _e183_ = select(_e171_, _e170_, _e172_);
                                let _e184_ = select(_e170_, _e171_, _e172_);
                                let _e185_ = bitcast<u32>(_e158_);
                                let _e186_ = (_e185_ & 2139095040u);
                                let _e187_ = (_e186_ == 2139095040u);
                                if _e187_ {
                                    if ((_e185_ & 2147483647u) > 2139095040u) {
                                        phi_3157_ = f32();
                                        phi_3158_ = true;
                                    } else {
                                        let _e218_ = ((_e185_ & 2147483648u) == 0u);
                                        if _e218_ {
                                            let _e219_ = bitcast<u32>(_e184_);
                                            let _e220_ = (_e219_ & 2139095040u);
                                            if (_e220_ == 2139095040u) {
                                                phi_3149_ = true;
                                            } else {
                                                phi_3149_ = (_e220_ == 0u);
                                            }
                                            let _e224_ = phi_3149_;
                                            phi_3155_ = bitcast<f32>(select(_e219_, 0u, _e224_));
                                        } else {
                                            phi_3155_ = f32();
                                        }
                                        let _e228_ = phi_3155_;
                                        phi_3157_ = _e228_;
                                        phi_3158_ = select(true, false, _e218_);
                                    }
                                    let _e231_ = phi_3157_;
                                    let _e233_ = phi_3158_;
                                    if _e233_ {
                                        let _e234_ = bitcast<u32>(_e183_);
                                        let _e235_ = (_e234_ & 2139095040u);
                                        if (_e235_ == 2139095040u) {
                                            phi_3172_ = true;
                                        } else {
                                            phi_3172_ = (_e235_ == 0u);
                                        }
                                        let _e239_ = phi_3172_;
                                        phi_3178_ = bitcast<f32>(select(_e234_, 0u, _e239_));
                                    } else {
                                        phi_3178_ = _e231_;
                                    }
                                    let _e243_ = phi_3178_;
                                    phi_3237_ = _e243_;
                                } else {
                                    if (_e158_ < _e183_) {
                                        let _e205_ = bitcast<u32>(_e183_);
                                        let _e206_ = (_e205_ & 2139095040u);
                                        if (_e206_ == 2139095040u) {
                                            phi_3193_ = true;
                                        } else {
                                            phi_3193_ = (_e206_ == 0u);
                                        }
                                        let _e210_ = phi_3193_;
                                        phi_3236_ = bitcast<f32>(select(_e205_, 0u, _e210_));
                                    } else {
                                        if (_e158_ > _e184_) {
                                            let _e195_ = bitcast<u32>(_e184_);
                                            let _e196_ = (_e195_ & 2139095040u);
                                            if (_e196_ == 2139095040u) {
                                                phi_3213_ = true;
                                            } else {
                                                phi_3213_ = (_e196_ == 0u);
                                            }
                                            let _e200_ = phi_3213_;
                                            phi_3235_ = bitcast<f32>(select(_e195_, 0u, _e200_));
                                        } else {
                                            if _e187_ {
                                                phi_3229_ = true;
                                            } else {
                                                phi_3229_ = (_e186_ == 0u);
                                            }
                                            let _e192_ = phi_3229_;
                                            phi_3235_ = bitcast<f32>(select(_e185_, 0u, _e192_));
                                        }
                                        let _e204_ = phi_3235_;
                                        phi_3236_ = _e204_;
                                    }
                                    let _e214_ = phi_3236_;
                                    phi_3237_ = _e214_;
                                }
                                let _e245_ = phi_3237_;
                                phi_3256_ = _e245_;
                            } else {
                                let _e175_ = bitcast<u32>((0.5f * (_e168_ + _e169_)));
                                let _e176_ = (_e175_ & 2139095040u);
                                if (_e176_ == 2139095040u) {
                                    phi_3250_ = true;
                                } else {
                                    phi_3250_ = (_e176_ == 0u);
                                }
                                let _e180_ = phi_3250_;
                                phi_3256_ = bitcast<f32>(select(_e175_, 0u, _e180_));
                            }
                            let _e247_ = phi_3256_;
                            if (_e55_ == 0u) {
                                phi_3267_ = 0u;
                            } else {
                                let _e249_ = (_e55_ - 1u);
                                phi_3267_ = select(_e249_, _e101_, (_e101_ < _e249_));
                            }
                            let _e253_ = phi_3267_;
                            if (_e253_ < _e55_) {
                            } else {
                                phi_3814_ = true;
                                break;
                            }
                            theta.member[_e253_] = _e247_;
                            phi_2981_ = (_e101_ + 256u);
                        } else {
                            phi_2981_ = u32();
                        }
                        let _e259_ = phi_2981_;
                        phi_2980_ = _e259_;
                        phi_3814_ = false;
                        if !(_e102_) {
                            break;
                        }
                    }
                    let _e262_ = phi_3814_;
                    if _e262_ {
                        break;
                    }
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn inner_opt_step(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    function_();
    return;
}

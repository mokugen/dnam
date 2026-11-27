struct type_6_ {
    member: array<f32>,
}

struct type_15_ {
    member: u32,
    member_1_: u32,
}

struct type_20_ {
    member: f32,
    member_1_: f32,
}

struct type_22_ {
    member: array<f32, 4>,
}

struct type_23_ {
    member: f32,
    member_1_: f32,
    member_2_: f32,
    member_3_: f32,
    member_4_: f32,
    member_5_: f32,
}

struct type_25_ {
    member: array<vec2<f32>>,
}

struct type_27_ {
    member: type_22_,
    member_1_: type_22_,
    member_2_: type_22_,
}

struct type_28_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: f32,
    member_5_: f32,
    member_6_: u32,
    member_7_: u32,
}

struct type_29_ {
    member: type_28_,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<workgroup> scan_scratch: array<f32, 1536>;
@group(0) @binding(2) 
var<storage> target_audio: type_25_;
@group(0) @binding(0) 
var<uniform> uniforms: type_29_;
@group(0) @binding(1) 
var<storage> input_audio: type_25_;
@group(0) @binding(3) 
var<storage> seed_params: type_6_;
@group(0) @binding(4) 
var<storage, read_write> seed_losses: type_6_;
var<workgroup> err_scratch: array<f32, 256>;
var<workgroup> tgt_scratch: array<f32, 256>;

fn function_() {
    var phi_222_: u32;
    var phi_229_: u32;
    var phi_252_: u32;
    var phi_275_: bool;
    var phi_291_: bool;
    var phi_299_: vec2<f32>;
    var phi_12004_: bool;
    var phi_302_: type_15_;
    var phi_305_: f32;
    var phi_307_: f32;
    var phi_303_: type_15_;
    var phi_330_: type_15_;
    var phi_348_: u32;
    var phi_359_: u32;
    var phi_371_: u32;
    var phi_390_: u32;
    var phi_423_: f32;
    var phi_429_: f32;
    var phi_8128_: bool;
    var phi_600_: bool;
    var phi_606_: f32;
    var phi_607_: f32;
    var phi_526_: f32;
    var phi_528_: f32;
    var phi_529_: bool;
    var phi_549_: f32;
    var phi_608_: f32;
    var phi_609_: f32;
    var phi_8188_: bool;
    var phi_695_: bool;
    var phi_8248_: bool;
    var phi_660_: bool;
    var phi_701_: f32;
    var phi_702_: f32;
    var phi_703_: f32;
    var phi_704_: bool;
    var phi_720_: u32;
    var phi_738_: bool;
    var phi_755_: u32;
    var phi_773_: bool;
    var phi_790_: u32;
    var phi_808_: bool;
    var phi_825_: u32;
    var phi_843_: bool;
    var phi_860_: u32;
    var phi_878_: bool;
    var phi_895_: u32;
    var phi_913_: bool;
    var phi_8308_: bool;
    var phi_8368_: bool;
    var phi_1494_: bool;
    var phi_1500_: f32;
    var phi_1501_: f32;
    var phi_1420_: f32;
    var phi_1422_: f32;
    var phi_1423_: bool;
    var phi_1502_: f32;
    var phi_1514_: u32;
    var phi_1532_: bool;
    var phi_1549_: u32;
    var phi_1567_: bool;
    var phi_1584_: u32;
    var phi_1602_: bool;
    var phi_1619_: u32;
    var phi_1637_: bool;
    var phi_1654_: u32;
    var phi_1778_: bool;
    var phi_1784_: f32;
    var phi_1785_: f32;
    var phi_1704_: f32;
    var phi_1706_: f32;
    var phi_1707_: bool;
    var phi_1727_: f32;
    var phi_1786_: f32;
    var phi_1798_: u32;
    var phi_1922_: bool;
    var phi_1928_: f32;
    var phi_1929_: f32;
    var phi_1848_: f32;
    var phi_1850_: f32;
    var phi_1851_: bool;
    var phi_1871_: f32;
    var phi_1930_: f32;
    var phi_1942_: u32;
    var phi_2066_: bool;
    var phi_2072_: f32;
    var phi_2073_: f32;
    var phi_1992_: f32;
    var phi_1994_: f32;
    var phi_1995_: bool;
    var phi_2015_: f32;
    var phi_2074_: f32;
    var phi_2086_: u32;
    var phi_2210_: bool;
    var phi_2216_: f32;
    var phi_2217_: f32;
    var phi_2136_: f32;
    var phi_2138_: f32;
    var phi_2139_: bool;
    var phi_2159_: f32;
    var phi_2218_: f32;
    var phi_2240_: type_23_;
    var phi_2300_: u32;
    var phi_2301_: u32;
    var phi_11808_: bool;
    var phi_2563_: u32;
    var phi_2564_: u32;
    var phi_12109_: bool;
    var phi_3106_: bool;
    var phi_3122_: bool;
    var phi_3130_: type_20_;
    var phi_3153_: type_23_;
    var phi_3213_: u32;
    var phi_3214_: u32;
    var phi_12124_: bool;
    var phi_3476_: u32;
    var phi_3477_: u32;
    var phi_12144_: bool;
    var phi_4019_: bool;
    var phi_4035_: bool;
    var phi_4043_: type_20_;
    var phi_4108_: type_15_;
    var phi_4111_: type_22_;
    var phi_4113_: type_22_;
    var phi_4109_: type_15_;
    var phi_4136_: type_15_;
    var phi_4189_: bool;
    var phi_4205_: bool;
    var phi_4221_: bool;
    var phi_4237_: bool;
    var phi_8428_: bool;
    var phi_8443_: bool;
    var phi_8458_: bool;
    var phi_8473_: bool;
    var phi_8488_: bool;
    var phi_8503_: bool;
    var phi_8518_: bool;
    var phi_8533_: bool;
    var phi_8548_: bool;
    var phi_8563_: bool;
    var phi_8578_: bool;
    var phi_8593_: bool;
    var phi_8608_: bool;
    var phi_8623_: bool;
    var phi_8638_: bool;
    var phi_8653_: bool;
    var phi_8668_: bool;
    var phi_8683_: bool;
    var phi_8698_: bool;
    var phi_8713_: bool;
    var phi_8728_: bool;
    var phi_8743_: bool;
    var phi_8758_: bool;
    var phi_8773_: bool;
    var phi_8788_: bool;
    var phi_8803_: bool;
    var phi_8818_: bool;
    var phi_8833_: bool;
    var phi_8848_: bool;
    var phi_8863_: bool;
    var phi_8878_: bool;
    var phi_8893_: bool;
    var phi_4945_: bool;
    var phi_4961_: bool;
    var phi_4977_: bool;
    var phi_4993_: bool;
    var phi_5015_: bool;
    var phi_5031_: bool;
    var phi_5047_: bool;
    var phi_5063_: bool;
    var phi_9028_: bool;
    var phi_9043_: bool;
    var phi_9058_: bool;
    var phi_9073_: bool;
    var phi_9088_: bool;
    var phi_9103_: bool;
    var phi_9118_: bool;
    var phi_9133_: bool;
    var phi_9148_: bool;
    var phi_9163_: bool;
    var phi_9178_: bool;
    var phi_9193_: bool;
    var phi_9208_: bool;
    var phi_9223_: bool;
    var phi_9238_: bool;
    var phi_9253_: bool;
    var phi_9268_: bool;
    var phi_9283_: bool;
    var phi_9298_: bool;
    var phi_9313_: bool;
    var phi_9328_: bool;
    var phi_9343_: bool;
    var phi_9358_: bool;
    var phi_9373_: bool;
    var phi_9388_: bool;
    var phi_9403_: bool;
    var phi_9418_: bool;
    var phi_9433_: bool;
    var phi_9448_: bool;
    var phi_9463_: bool;
    var phi_9478_: bool;
    var phi_9493_: bool;
    var phi_5771_: bool;
    var phi_5787_: bool;
    var phi_5803_: bool;
    var phi_5819_: bool;
    var phi_4112_: type_22_;
    var phi_4114_: type_22_;
    var phi_12159_: bool;
    var phi_12841_: bool;
    var phi_5907_: f32;
    var phi_5919_: f32;
    var phi_9628_: bool;
    var phi_9643_: bool;
    var phi_5943_: bool;
    var phi_5949_: f32;
    var phi_5967_: bool;
    var phi_5973_: f32;
    var phi_6002_: f32;
    var phi_6004_: f32;
    var phi_6005_: bool;
    var phi_6026_: f32;
    var phi_6028_: f32;
    var phi_6029_: bool;
    var phi_9658_: bool;
    var phi_9673_: bool;
    var phi_9688_: bool;
    var phi_9703_: bool;
    var phi_9718_: bool;
    var phi_9733_: bool;
    var phi_9748_: bool;
    var phi_9763_: bool;
    var phi_9778_: bool;
    var phi_9793_: bool;
    var phi_9808_: bool;
    var phi_9823_: bool;
    var phi_9838_: bool;
    var phi_9853_: bool;
    var phi_9868_: bool;
    var phi_9883_: bool;
    var phi_9898_: bool;
    var phi_9913_: bool;
    var phi_9928_: bool;
    var phi_9943_: bool;
    var phi_9958_: bool;
    var phi_9973_: bool;
    var phi_9988_: bool;
    var phi_10003_: bool;
    var phi_10018_: bool;
    var phi_10033_: bool;
    var phi_10048_: bool;
    var phi_10063_: bool;
    var phi_10078_: bool;
    var phi_10093_: bool;
    var phi_10108_: bool;
    var phi_10123_: bool;
    var phi_10138_: bool;
    var phi_10153_: bool;
    var phi_10168_: bool;
    var phi_10183_: bool;
    var phi_10198_: bool;
    var phi_10213_: bool;
    var phi_10228_: bool;
    var phi_10243_: bool;
    var phi_10258_: bool;
    var phi_10273_: bool;
    var phi_10288_: bool;
    var phi_10303_: bool;
    var phi_10318_: bool;
    var phi_10333_: bool;
    var phi_10348_: bool;
    var phi_10363_: bool;
    var phi_10378_: bool;
    var phi_10393_: bool;
    var phi_10408_: bool;
    var phi_10423_: bool;
    var phi_10438_: bool;
    var phi_10453_: bool;
    var phi_10468_: bool;
    var phi_10483_: bool;
    var phi_10498_: bool;
    var phi_10513_: bool;
    var phi_10528_: bool;
    var phi_10543_: bool;
    var phi_10558_: bool;
    var phi_10573_: bool;
    var phi_10588_: bool;
    var phi_10603_: bool;
    var phi_6456_: bool;
    var phi_6458_: bool;
    var phi_6459_: bool;
    var phi_6460_: bool;
    var phi_6461_: bool;
    var phi_10618_: bool;
    var phi_10633_: bool;
    var phi_10648_: bool;
    var phi_10663_: bool;
    var phi_6839_: type_27_;
    var phi_10678_: bool;
    var phi_10693_: bool;
    var phi_10708_: bool;
    var phi_10723_: bool;
    var phi_10738_: bool;
    var phi_10753_: bool;
    var phi_10768_: bool;
    var phi_10783_: bool;
    var phi_10798_: bool;
    var phi_10813_: bool;
    var phi_10828_: bool;
    var phi_10843_: bool;
    var phi_10858_: bool;
    var phi_10873_: bool;
    var phi_10888_: bool;
    var phi_10903_: bool;
    var phi_10918_: bool;
    var phi_10933_: bool;
    var phi_10948_: bool;
    var phi_10963_: bool;
    var phi_10978_: bool;
    var phi_10993_: bool;
    var phi_11008_: bool;
    var phi_11023_: bool;
    var phi_11038_: bool;
    var phi_11053_: bool;
    var phi_11068_: bool;
    var phi_11083_: bool;
    var phi_11098_: bool;
    var phi_11113_: bool;
    var phi_11128_: bool;
    var phi_11143_: bool;
    var phi_11158_: bool;
    var phi_11173_: bool;
    var phi_11188_: bool;
    var phi_11203_: bool;
    var phi_11218_: bool;
    var phi_11233_: bool;
    var phi_11248_: bool;
    var phi_11263_: bool;
    var phi_11278_: bool;
    var phi_11293_: bool;
    var phi_11308_: bool;
    var phi_11323_: bool;
    var phi_11338_: bool;
    var phi_11353_: bool;
    var phi_11368_: bool;
    var phi_11383_: bool;
    var phi_11398_: bool;
    var phi_11413_: bool;
    var phi_11428_: bool;
    var phi_11443_: bool;
    var phi_11458_: bool;
    var phi_11473_: bool;
    var phi_11488_: bool;
    var phi_11503_: bool;
    var phi_11518_: bool;
    var phi_11533_: bool;
    var phi_11548_: bool;
    var phi_11563_: bool;
    var phi_11578_: bool;
    var phi_11593_: bool;
    var phi_11608_: bool;
    var phi_11623_: bool;
    var phi_7263_: bool;
    var phi_7265_: bool;
    var phi_7266_: bool;
    var phi_7267_: bool;
    var phi_7268_: bool;
    var phi_11638_: bool;
    var phi_11653_: bool;
    var phi_11668_: bool;
    var phi_11683_: bool;
    var phi_7646_: type_27_;
    var phi_7655_: f32;
    var phi_7656_: f32;
    var phi_12445_: bool;
    var phi_306_: f32;
    var phi_308_: f32;
    var phi_12444_: bool;
    var phi_7670_: u32;
    var phi_7693_: bool;
    var phi_7709_: bool;
    var phi_7717_: vec2<f32>;
    var local: f32;
    var phi_7723_: f32;
    var local_1_: f32;
    var phi_7729_: f32;
    var phi_7746_: bool;
    var phi_7752_: f32;
    var phi_7773_: bool;
    var phi_7779_: f32;
    var phi_7790_: bool;
    var phi_7806_: bool;
    var phi_7836_: u32;
    var phi_7837_: u32;
    var phi_12852_: bool;
    var phi_7897_: bool;
    var phi_7915_: bool;
    var phi_7955_: f32;
    var phi_7956_: bool;
    var phi_11698_: bool;
    var phi_8047_: bool;
    var phi_11758_: bool;
    var phi_8012_: bool;
    var phi_8053_: f32;
    var phi_8054_: f32;
    var phi_8055_: f32;
    var phi_8056_: bool;
    var phi_8072_: bool;
    var phi_8092_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e112_ = wg_id_1_;
            let _e113_ = lid_1_;
            let _e115_ = arrayLength((&input_audio.member));
            let _e117_ = arrayLength((&target_audio.member));
            let _e119_ = arrayLength((&seed_params.member));
            let _e121_ = arrayLength((&seed_losses.member));
            let _e127_ = uniforms.member.member;
            let _e129_ = select(1u, _e127_, (1u < _e127_));
            let _e132_ = uniforms.member.member_1_;
            let _e135_ = uniforms.member.member_2_;
            let _e137_ = select(1u, _e135_, (1u < _e135_));
            let _e140_ = uniforms.member.member_3_;
            if (_e132_ == 0u) {
                phi_222_ = 1u;
            } else {
                phi_222_ = (((_e132_ - 1u) >> bitcast<u32>(8i)) + 1u);
            }
            let _e149_ = phi_222_;
            let _e150_ = (_e112_.y * 256u);
            if (_e132_ > _e150_) {
                phi_229_ = (_e132_ - _e150_);
            } else {
                phi_229_ = 0u;
            }
            let _e154_ = phi_229_;
            let _e157_ = (_e150_ + _e113_);
            let _e158_ = (_e157_ < _e132_);
            let _e160_ = ((_e112_.z * _e132_) + _e157_);
            if _e158_ {
                if (_e115_ == 0u) {
                    phi_252_ = 0u;
                } else {
                    let _e162_ = (_e115_ - 1u);
                    phi_252_ = select(_e162_, _e160_, (_e160_ < _e162_));
                }
                let _e166_ = phi_252_;
                if (_e166_ < _e115_) {
                } else {
                    break;
                }
                let _e171_ = input_audio.member[_e166_].x;
                let _e175_ = input_audio.member[_e166_].y;
                let _e176_ = bitcast<u32>(_e171_);
                let _e177_ = (_e176_ & 2139095040u);
                if (_e177_ == 2139095040u) {
                    phi_275_ = true;
                } else {
                    phi_275_ = (_e177_ == 0u);
                }
                let _e181_ = phi_275_;
                let _e184_ = bitcast<u32>(_e175_);
                let _e185_ = (_e184_ & 2139095040u);
                if (_e185_ == 2139095040u) {
                    phi_291_ = true;
                } else {
                    phi_291_ = (_e185_ == 0u);
                }
                let _e189_ = phi_291_;
                phi_299_ = vec2<f32>(bitcast<f32>(select(_e176_, 0u, _e181_)), bitcast<f32>(select(_e184_, 0u, _e189_)));
            } else {
                phi_299_ = vec2<f32>(0f, 0f);
            }
            let _e194_ = phi_299_;
            phi_12004_ = false;
            phi_302_ = type_15_(0u, _e137_);
            phi_305_ = 0f;
            phi_307_ = 0f;
            loop {
                let _e197_ = phi_12004_;
                let _e199_ = phi_302_;
                let _e201_ = phi_305_;
                let _e203_ = phi_307_;
                local = _e203_;
                local_1_ = _e201_;
                if (_e199_.member < _e199_.member_1_) {
                    phi_303_ = type_15_((_e199_.member + 1u), _e199_.member_1_);
                    phi_330_ = type_15_(1u, _e199_.member);
                } else {
                    phi_303_ = _e199_;
                    phi_330_ = type_15_(0u, u32());
                }
                let _e216_ = phi_303_;
                let _e218_ = phi_330_;
                let _e222_ = (bitcast<i32>(_e218_.member) != 0i);
                if _e222_ {
                    if (_e129_ == 0u) {
                        phi_348_ = 0u;
                    } else {
                        let _e224_ = (_e129_ - 1u);
                        phi_348_ = select(_e224_, _e112_.x, (_e112_.x < _e224_));
                    }
                    let _e228_ = phi_348_;
                    if (_e119_ >= ((_e129_ * _e137_) * 16u)) {
                        phi_359_ = (((_e228_ * _e137_) + _e218_.member_1_) * 16u);
                    } else {
                        phi_359_ = (_e228_ * 16u);
                    }
                    let _e237_ = phi_359_;
                    let _e238_ = (_e119_ == 0u);
                    if _e238_ {
                        phi_371_ = 0u;
                    } else {
                        let _e239_ = (_e119_ - 1u);
                        phi_371_ = select(_e239_, _e237_, (_e237_ < _e239_));
                    }
                    let _e243_ = phi_371_;
                    if (_e243_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e247_ = seed_params.member[_e243_];
                    let _e248_ = (_e237_ + 1u);
                    if _e238_ {
                        phi_390_ = 0u;
                    } else {
                        let _e249_ = (_e119_ - 1u);
                        phi_390_ = select(_e249_, _e248_, (_e248_ < _e249_));
                    }
                    let _e253_ = phi_390_;
                    if (_e253_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e257_ = seed_params.member[_e253_];
                    let _e261_ = select(0.001f, _e247_, ((bitcast<u32>(_e247_) & 2139095040u) != 2139095040u));
                    if (_e261_ != _e261_) {
                        phi_423_ = 0.00001f;
                    } else {
                        let _e264_ = select(_e261_, 0.00001f, (_e261_ < 0.00001f));
                        phi_423_ = select(_e264_, 0.495f, (_e264_ > 0.495f));
                    }
                    let _e268_ = phi_423_;
                    let _e269_ = (_e268_ <= 0.25f);
                    if _e269_ {
                        phi_429_ = _e268_;
                    } else {
                        phi_429_ = (0.5f - _e268_);
                    }
                    let _e272_ = phi_429_;
                    let _e273_ = (3.1415927f * _e272_);
                    let _e274_ = (_e273_ * _e273_);
                    let _e285_ = (_e273_ * (1f + (_e274_ * (-0.16666667f + (_e274_ * (0.008333334f + (_e274_ * (-0.0001984127f + (_e274_ * (0.0000027557319f + (_e274_ * -0.000000025052108f)))))))))));
                    let _e295_ = (1f + (_e274_ * (-0.5f + (_e274_ * (0.041666668f + (_e274_ * (-0.0013888889f + (_e274_ * (0.000024801588f + (_e274_ * -0.0000002755732f))))))))));
                    let _e297_ = select(_e285_, _e295_, _e269_);
                    if (_e297_ != _e297_) {
                        phi_8128_ = true;
                    } else {
                        phi_8128_ = (0.000001f >= _e297_);
                    }
                    let _e301_ = phi_8128_;
                    let _e302_ = select(_e297_, 0.000001f, _e301_);
                    let _e305_ = bitcast<f32>((2129859011u - bitcast<u32>(_e302_)));
                    let _e309_ = (_e305_ + (_e305_ * (1f - (_e302_ * _e305_))));
                    let _e313_ = (_e309_ + (_e309_ * (1f - (_e302_ * _e309_))));
                    let _e318_ = (select(_e295_, _e285_, _e269_) * (_e313_ + (_e313_ * (1f - (_e302_ * _e313_)))));
                    let _e319_ = bitcast<u32>(_e257_);
                    let _e320_ = (_e319_ & 2139095040u);
                    if (_e320_ != 2139095040u) {
                        let _e322_ = (_e320_ == 2139095040u);
                        if _e322_ {
                            if ((_e319_ & 2147483647u) > 2139095040u) {
                                phi_528_ = f32();
                                phi_529_ = true;
                            } else {
                                let _e337_ = ((_e319_ & 2147483648u) == 0u);
                                if _e337_ {
                                    phi_526_ = 64f;
                                } else {
                                    phi_526_ = f32();
                                }
                                let _e339_ = phi_526_;
                                phi_528_ = _e339_;
                                phi_529_ = select(true, false, _e337_);
                            }
                            let _e342_ = phi_528_;
                            let _e344_ = phi_529_;
                            if _e344_ {
                                phi_549_ = 0.05f;
                            } else {
                                phi_549_ = _e342_;
                            }
                            let _e346_ = phi_549_;
                            phi_608_ = _e346_;
                        } else {
                            if (_e257_ < 0.05f) {
                                phi_607_ = 0.05f;
                            } else {
                                if (_e257_ > 64f) {
                                    phi_606_ = 64f;
                                } else {
                                    if _e322_ {
                                        phi_600_ = true;
                                    } else {
                                        phi_600_ = (_e320_ == 0u);
                                    }
                                    let _e327_ = phi_600_;
                                    phi_606_ = bitcast<f32>(select(_e319_, 0u, _e327_));
                                }
                                let _e331_ = phi_606_;
                                phi_607_ = _e331_;
                            }
                            let _e333_ = phi_607_;
                            phi_608_ = _e333_;
                        }
                        let _e348_ = phi_608_;
                        phi_609_ = _e348_;
                    } else {
                        phi_609_ = 0.70710677f;
                    }
                    let _e350_ = phi_609_;
                    if ((bitcast<u32>(_e350_) & 2139095040u) != 2139095040u) {
                        let _e354_ = (_e350_ <= 0f);
                        if _e354_ {
                            phi_702_ = f32();
                        } else {
                            if (_e350_ < 0.00001f) {
                                let _e386_ = (select(0.000000000000000000000001f, _e350_, (_e350_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                if (_e386_ != _e386_) {
                                    phi_8248_ = true;
                                } else {
                                    phi_8248_ = (0.000001f >= _e386_);
                                }
                                let _e390_ = phi_8248_;
                                let _e391_ = select(_e386_, 0.000001f, _e390_);
                                let _e394_ = bitcast<f32>((2129859011u - bitcast<u32>(_e391_)));
                                let _e398_ = (_e394_ + (_e394_ * (1f - (_e391_ * _e394_))));
                                let _e402_ = (_e398_ + (_e398_ * (1f - (_e391_ * _e398_))));
                                let _e408_ = bitcast<u32>(((_e402_ + (_e402_ * (1f - (_e391_ * _e402_)))) * 18446744000000000000f));
                                let _e409_ = (_e408_ & 2139095040u);
                                if (_e409_ == 2139095040u) {
                                    phi_660_ = true;
                                } else {
                                    phi_660_ = (_e409_ == 0u);
                                }
                                let _e413_ = phi_660_;
                                phi_701_ = bitcast<f32>(select(_e408_, 0u, _e413_));
                            } else {
                                if (_e350_ != _e350_) {
                                    phi_8188_ = true;
                                } else {
                                    phi_8188_ = (0.000001f >= _e350_);
                                }
                                let _e359_ = phi_8188_;
                                let _e360_ = select(_e350_, 0.000001f, _e359_);
                                let _e363_ = bitcast<f32>((2129859011u - bitcast<u32>(_e360_)));
                                let _e367_ = (_e363_ + (_e363_ * (1f - (_e360_ * _e363_))));
                                let _e371_ = (_e367_ + (_e367_ * (1f - (_e360_ * _e367_))));
                                let _e376_ = bitcast<u32>((_e371_ + (_e371_ * (1f - (_e360_ * _e371_)))));
                                let _e377_ = (_e376_ & 2139095040u);
                                if (_e377_ == 2139095040u) {
                                    phi_695_ = true;
                                } else {
                                    phi_695_ = (_e377_ == 0u);
                                }
                                let _e381_ = phi_695_;
                                phi_701_ = bitcast<f32>(select(_e376_, 0u, _e381_));
                            }
                            let _e417_ = phi_701_;
                            phi_702_ = _e417_;
                        }
                        let _e419_ = phi_702_;
                        phi_703_ = _e419_;
                        phi_704_ = _e354_;
                    } else {
                        phi_703_ = f32();
                        phi_704_ = true;
                    }
                    let _e421_ = phi_703_;
                    let _e423_ = phi_704_;
                    let _e424_ = select(_e421_, 0f, _e423_);
                    let _e425_ = (_e237_ + 6u);
                    if _e238_ {
                        phi_720_ = 0u;
                    } else {
                        let _e426_ = (_e119_ - 1u);
                        phi_720_ = select(_e426_, _e425_, (_e425_ < _e426_));
                    }
                    let _e430_ = phi_720_;
                    if (_e430_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e434_ = seed_params.member[_e430_];
                    let _e435_ = bitcast<u32>(_e434_);
                    let _e436_ = (_e435_ & 2139095040u);
                    if (_e436_ == 2139095040u) {
                        phi_738_ = true;
                    } else {
                        phi_738_ = (_e436_ == 0u);
                    }
                    let _e440_ = phi_738_;
                    let _e442_ = bitcast<f32>(select(_e435_, 0u, _e440_));
                    let _e443_ = (_e237_ + 7u);
                    if _e238_ {
                        phi_755_ = 0u;
                    } else {
                        let _e444_ = (_e119_ - 1u);
                        phi_755_ = select(_e444_, _e443_, (_e443_ < _e444_));
                    }
                    let _e448_ = phi_755_;
                    if (_e448_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e452_ = seed_params.member[_e448_];
                    let _e453_ = bitcast<u32>(_e452_);
                    let _e454_ = (_e453_ & 2139095040u);
                    if (_e454_ == 2139095040u) {
                        phi_773_ = true;
                    } else {
                        phi_773_ = (_e454_ == 0u);
                    }
                    let _e458_ = phi_773_;
                    let _e460_ = bitcast<f32>(select(_e453_, 0u, _e458_));
                    let _e461_ = (_e237_ + 8u);
                    if _e238_ {
                        phi_790_ = 0u;
                    } else {
                        let _e462_ = (_e119_ - 1u);
                        phi_790_ = select(_e462_, _e461_, (_e461_ < _e462_));
                    }
                    let _e466_ = phi_790_;
                    if (_e466_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e470_ = seed_params.member[_e466_];
                    let _e471_ = bitcast<u32>(_e470_);
                    let _e472_ = (_e471_ & 2139095040u);
                    if (_e472_ == 2139095040u) {
                        phi_808_ = true;
                    } else {
                        phi_808_ = (_e472_ == 0u);
                    }
                    let _e476_ = phi_808_;
                    let _e478_ = bitcast<f32>(select(_e471_, 0u, _e476_));
                    let _e479_ = (_e237_ + 9u);
                    if _e238_ {
                        phi_825_ = 0u;
                    } else {
                        let _e480_ = (_e119_ - 1u);
                        phi_825_ = select(_e480_, _e479_, (_e479_ < _e480_));
                    }
                    let _e484_ = phi_825_;
                    if (_e484_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e488_ = seed_params.member[_e484_];
                    let _e489_ = bitcast<u32>(_e488_);
                    let _e490_ = (_e489_ & 2139095040u);
                    if (_e490_ == 2139095040u) {
                        phi_843_ = true;
                    } else {
                        phi_843_ = (_e490_ == 0u);
                    }
                    let _e494_ = phi_843_;
                    let _e496_ = bitcast<f32>(select(_e489_, 0u, _e494_));
                    let _e497_ = (_e237_ + 10u);
                    if _e238_ {
                        phi_860_ = 0u;
                    } else {
                        let _e498_ = (_e119_ - 1u);
                        phi_860_ = select(_e498_, _e497_, (_e497_ < _e498_));
                    }
                    let _e502_ = phi_860_;
                    if (_e502_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e506_ = seed_params.member[_e502_];
                    let _e507_ = bitcast<u32>(_e506_);
                    let _e508_ = (_e507_ & 2139095040u);
                    if (_e508_ == 2139095040u) {
                        phi_878_ = true;
                    } else {
                        phi_878_ = (_e508_ == 0u);
                    }
                    let _e512_ = phi_878_;
                    let _e514_ = bitcast<f32>(select(_e507_, 0u, _e512_));
                    let _e515_ = (_e237_ + 11u);
                    if _e238_ {
                        phi_895_ = 0u;
                    } else {
                        let _e516_ = (_e119_ - 1u);
                        phi_895_ = select(_e516_, _e515_, (_e515_ < _e516_));
                    }
                    let _e520_ = phi_895_;
                    if (_e520_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e524_ = seed_params.member[_e520_];
                    let _e525_ = bitcast<u32>(_e524_);
                    let _e526_ = (_e525_ & 2139095040u);
                    if (_e526_ == 2139095040u) {
                        phi_913_ = true;
                    } else {
                        phi_913_ = (_e526_ == 0u);
                    }
                    let _e530_ = phi_913_;
                    let _e532_ = bitcast<f32>(select(_e525_, 0u, _e530_));
                    let _e538_ = select(select(select(0f, 0.999995f, (0.999995f <= _e442_)), _e442_, (_e442_ <= 0.999995f)), 0f, (_e442_ <= 0f));
                    let _e556_ = select(select(select(0f, 64f, (64f <= _e496_)), _e496_, (_e496_ <= 64f)), 0f, (_e496_ <= 0f));
                    let _e562_ = select(select(select(0f, 1f, (1f <= _e514_)), _e514_, (_e514_ <= 1f)), 0f, (_e514_ <= 0f));
                    let _e568_ = select(select(select(0f, 1f, (1f <= _e532_)), _e532_, (_e532_ <= 1f)), 0f, (_e532_ <= 0f));
                    if (_e318_ != _e318_) {
                        phi_8308_ = true;
                    } else {
                        phi_8308_ = (0.00001f >= _e318_);
                    }
                    let _e572_ = phi_8308_;
                    let _e573_ = select(_e318_, 0.00001f, _e572_);
                    if (_e424_ != _e424_) {
                        phi_8368_ = true;
                    } else {
                        phi_8368_ = (0f >= _e424_);
                    }
                    let _e577_ = phi_8368_;
                    let _e578_ = select(_e424_, 0f, _e577_);
                    let _e581_ = (1f + (_e573_ * (_e573_ + _e578_)));
                    let _e584_ = bitcast<f32>((2129859011u - bitcast<u32>(_e581_)));
                    let _e588_ = (_e584_ + (_e584_ * (1f - (_e581_ * _e584_))));
                    let _e592_ = (_e588_ + (_e588_ * (1f - (_e581_ * _e588_))));
                    let _e593_ = (_e573_ * _e592_);
                    let _e594_ = (_e573_ * _e593_);
                    let _e595_ = (_e318_ + _e318_);
                    let _e596_ = bitcast<u32>(_e595_);
                    let _e597_ = (_e596_ & 2139095040u);
                    let _e598_ = (_e597_ == 2139095040u);
                    if _e598_ {
                        if ((_e596_ & 2147483647u) > 2139095040u) {
                            phi_1422_ = f32();
                            phi_1423_ = true;
                        } else {
                            let _e613_ = ((_e596_ & 2147483648u) == 0u);
                            if _e613_ {
                                phi_1420_ = 128f;
                            } else {
                                phi_1420_ = f32();
                            }
                            let _e615_ = phi_1420_;
                            phi_1422_ = _e615_;
                            phi_1423_ = select(true, false, _e613_);
                        }
                        let _e618_ = phi_1422_;
                        let _e620_ = phi_1423_;
                        phi_1502_ = select(_e618_, 0f, _e620_);
                    } else {
                        if (_e595_ < 0f) {
                            phi_1501_ = 0f;
                        } else {
                            if (_e595_ > 128f) {
                                phi_1500_ = 128f;
                            } else {
                                if _e598_ {
                                    phi_1494_ = true;
                                } else {
                                    phi_1494_ = (_e597_ == 0u);
                                }
                                let _e603_ = phi_1494_;
                                phi_1500_ = bitcast<f32>(select(_e596_, 0u, _e603_));
                            }
                            let _e607_ = phi_1500_;
                            phi_1501_ = _e607_;
                        }
                        let _e609_ = phi_1501_;
                        phi_1502_ = _e609_;
                    }
                    let _e623_ = phi_1502_;
                    let _e624_ = (_e237_ + 2u);
                    if _e238_ {
                        phi_1514_ = 0u;
                    } else {
                        let _e625_ = (_e119_ - 1u);
                        phi_1514_ = select(_e625_, _e624_, (_e624_ < _e625_));
                    }
                    let _e629_ = phi_1514_;
                    if (_e629_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e633_ = seed_params.member[_e629_];
                    let _e634_ = bitcast<u32>(_e633_);
                    let _e635_ = (_e634_ & 2139095040u);
                    if (_e635_ == 2139095040u) {
                        phi_1532_ = true;
                    } else {
                        phi_1532_ = (_e635_ == 0u);
                    }
                    let _e639_ = phi_1532_;
                    let _e641_ = bitcast<f32>(select(_e634_, 0u, _e639_));
                    let _e642_ = (_e237_ + 3u);
                    if _e238_ {
                        phi_1549_ = 0u;
                    } else {
                        let _e643_ = (_e119_ - 1u);
                        phi_1549_ = select(_e643_, _e642_, (_e642_ < _e643_));
                    }
                    let _e647_ = phi_1549_;
                    if (_e647_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e651_ = seed_params.member[_e647_];
                    let _e652_ = bitcast<u32>(_e651_);
                    let _e653_ = (_e652_ & 2139095040u);
                    if (_e653_ == 2139095040u) {
                        phi_1567_ = true;
                    } else {
                        phi_1567_ = (_e653_ == 0u);
                    }
                    let _e657_ = phi_1567_;
                    let _e659_ = bitcast<f32>(select(_e652_, 0u, _e657_));
                    let _e660_ = (_e237_ + 4u);
                    if _e238_ {
                        phi_1584_ = 0u;
                    } else {
                        let _e661_ = (_e119_ - 1u);
                        phi_1584_ = select(_e661_, _e660_, (_e660_ < _e661_));
                    }
                    let _e665_ = phi_1584_;
                    if (_e665_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e669_ = seed_params.member[_e665_];
                    let _e670_ = bitcast<u32>(_e669_);
                    let _e671_ = (_e670_ & 2139095040u);
                    if (_e671_ == 2139095040u) {
                        phi_1602_ = true;
                    } else {
                        phi_1602_ = (_e671_ == 0u);
                    }
                    let _e675_ = phi_1602_;
                    let _e677_ = bitcast<f32>(select(_e670_, 0u, _e675_));
                    let _e678_ = (_e237_ + 5u);
                    if _e238_ {
                        phi_1619_ = 0u;
                    } else {
                        let _e679_ = (_e119_ - 1u);
                        phi_1619_ = select(_e679_, _e678_, (_e678_ < _e679_));
                    }
                    let _e683_ = phi_1619_;
                    if (_e683_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e687_ = seed_params.member[_e683_];
                    let _e688_ = bitcast<u32>(_e687_);
                    let _e689_ = (_e688_ & 2139095040u);
                    if (_e689_ == 2139095040u) {
                        phi_1637_ = true;
                    } else {
                        phi_1637_ = (_e689_ == 0u);
                    }
                    let _e693_ = phi_1637_;
                    let _e695_ = bitcast<f32>(select(_e688_, 0u, _e693_));
                    let _e696_ = (_e237_ + 12u);
                    if _e238_ {
                        phi_1654_ = 0u;
                    } else {
                        let _e697_ = (_e119_ - 1u);
                        phi_1654_ = select(_e697_, _e696_, (_e696_ < _e697_));
                    }
                    let _e701_ = phi_1654_;
                    if (_e701_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e705_ = seed_params.member[_e701_];
                    let _e706_ = bitcast<u32>(_e705_);
                    let _e707_ = (_e706_ & 2139095040u);
                    let _e708_ = (_e707_ == 2139095040u);
                    if _e708_ {
                        if ((_e706_ & 2147483647u) > 2139095040u) {
                            phi_1706_ = f32();
                            phi_1707_ = true;
                        } else {
                            let _e723_ = ((_e706_ & 2147483648u) == 0u);
                            if _e723_ {
                                phi_1704_ = 1f;
                            } else {
                                phi_1704_ = f32();
                            }
                            let _e725_ = phi_1704_;
                            phi_1706_ = _e725_;
                            phi_1707_ = select(true, false, _e723_);
                        }
                        let _e728_ = phi_1706_;
                        let _e730_ = phi_1707_;
                        if _e730_ {
                            phi_1727_ = -1f;
                        } else {
                            phi_1727_ = _e728_;
                        }
                        let _e732_ = phi_1727_;
                        phi_1786_ = _e732_;
                    } else {
                        if (_e705_ < -1f) {
                            phi_1785_ = -1f;
                        } else {
                            if (_e705_ > 1f) {
                                phi_1784_ = 1f;
                            } else {
                                if _e708_ {
                                    phi_1778_ = true;
                                } else {
                                    phi_1778_ = (_e707_ == 0u);
                                }
                                let _e713_ = phi_1778_;
                                phi_1784_ = bitcast<f32>(select(_e706_, 0u, _e713_));
                            }
                            let _e717_ = phi_1784_;
                            phi_1785_ = _e717_;
                        }
                        let _e719_ = phi_1785_;
                        phi_1786_ = _e719_;
                    }
                    let _e734_ = phi_1786_;
                    let _e735_ = (_e237_ + 13u);
                    if _e238_ {
                        phi_1798_ = 0u;
                    } else {
                        let _e736_ = (_e119_ - 1u);
                        phi_1798_ = select(_e736_, _e735_, (_e735_ < _e736_));
                    }
                    let _e740_ = phi_1798_;
                    if (_e740_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e744_ = seed_params.member[_e740_];
                    let _e745_ = bitcast<u32>(_e744_);
                    let _e746_ = (_e745_ & 2139095040u);
                    let _e747_ = (_e746_ == 2139095040u);
                    if _e747_ {
                        if ((_e745_ & 2147483647u) > 2139095040u) {
                            phi_1850_ = f32();
                            phi_1851_ = true;
                        } else {
                            let _e762_ = ((_e745_ & 2147483648u) == 0u);
                            if _e762_ {
                                phi_1848_ = 4f;
                            } else {
                                phi_1848_ = f32();
                            }
                            let _e764_ = phi_1848_;
                            phi_1850_ = _e764_;
                            phi_1851_ = select(true, false, _e762_);
                        }
                        let _e767_ = phi_1850_;
                        let _e769_ = phi_1851_;
                        if _e769_ {
                            phi_1871_ = -4f;
                        } else {
                            phi_1871_ = _e767_;
                        }
                        let _e771_ = phi_1871_;
                        phi_1930_ = _e771_;
                    } else {
                        if (_e744_ < -4f) {
                            phi_1929_ = -4f;
                        } else {
                            if (_e744_ > 4f) {
                                phi_1928_ = 4f;
                            } else {
                                if _e747_ {
                                    phi_1922_ = true;
                                } else {
                                    phi_1922_ = (_e746_ == 0u);
                                }
                                let _e752_ = phi_1922_;
                                phi_1928_ = bitcast<f32>(select(_e745_, 0u, _e752_));
                            }
                            let _e756_ = phi_1928_;
                            phi_1929_ = _e756_;
                        }
                        let _e758_ = phi_1929_;
                        phi_1930_ = _e758_;
                    }
                    let _e773_ = phi_1930_;
                    let _e774_ = (_e237_ + 14u);
                    if _e238_ {
                        phi_1942_ = 0u;
                    } else {
                        let _e775_ = (_e119_ - 1u);
                        phi_1942_ = select(_e775_, _e774_, (_e774_ < _e775_));
                    }
                    let _e779_ = phi_1942_;
                    if (_e779_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e783_ = seed_params.member[_e779_];
                    let _e784_ = bitcast<u32>(_e783_);
                    let _e785_ = (_e784_ & 2139095040u);
                    let _e786_ = (_e785_ == 2139095040u);
                    if _e786_ {
                        if ((_e784_ & 2147483647u) > 2139095040u) {
                            phi_1994_ = f32();
                            phi_1995_ = true;
                        } else {
                            let _e801_ = ((_e784_ & 2147483648u) == 0u);
                            if _e801_ {
                                phi_1992_ = 64f;
                            } else {
                                phi_1992_ = f32();
                            }
                            let _e803_ = phi_1992_;
                            phi_1994_ = _e803_;
                            phi_1995_ = select(true, false, _e801_);
                        }
                        let _e806_ = phi_1994_;
                        let _e808_ = phi_1995_;
                        if _e808_ {
                            phi_2015_ = -64f;
                        } else {
                            phi_2015_ = _e806_;
                        }
                        let _e810_ = phi_2015_;
                        phi_2074_ = _e810_;
                    } else {
                        if (_e783_ < -64f) {
                            phi_2073_ = -64f;
                        } else {
                            if (_e783_ > 64f) {
                                phi_2072_ = 64f;
                            } else {
                                if _e786_ {
                                    phi_2066_ = true;
                                } else {
                                    phi_2066_ = (_e785_ == 0u);
                                }
                                let _e791_ = phi_2066_;
                                phi_2072_ = bitcast<f32>(select(_e784_, 0u, _e791_));
                            }
                            let _e795_ = phi_2072_;
                            phi_2073_ = _e795_;
                        }
                        let _e797_ = phi_2073_;
                        phi_2074_ = _e797_;
                    }
                    let _e812_ = phi_2074_;
                    let _e813_ = (_e237_ + 15u);
                    if _e238_ {
                        phi_2086_ = 0u;
                    } else {
                        let _e814_ = (_e119_ - 1u);
                        phi_2086_ = select(_e814_, _e813_, (_e813_ < _e814_));
                    }
                    let _e818_ = phi_2086_;
                    if (_e818_ < _e119_) {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e822_ = seed_params.member[_e818_];
                    let _e823_ = bitcast<u32>(_e822_);
                    let _e824_ = (_e823_ & 2139095040u);
                    let _e825_ = (_e824_ == 2139095040u);
                    if _e825_ {
                        if ((_e823_ & 2147483647u) > 2139095040u) {
                            phi_2138_ = f32();
                            phi_2139_ = true;
                        } else {
                            let _e840_ = ((_e823_ & 2147483648u) == 0u);
                            if _e840_ {
                                phi_2136_ = 64f;
                            } else {
                                phi_2136_ = f32();
                            }
                            let _e842_ = phi_2136_;
                            phi_2138_ = _e842_;
                            phi_2139_ = select(true, false, _e840_);
                        }
                        let _e845_ = phi_2138_;
                        let _e847_ = phi_2139_;
                        if _e847_ {
                            phi_2159_ = -64f;
                        } else {
                            phi_2159_ = _e845_;
                        }
                        let _e849_ = phi_2159_;
                        phi_2218_ = _e849_;
                    } else {
                        if (_e822_ < -64f) {
                            phi_2217_ = -64f;
                        } else {
                            if (_e822_ > 64f) {
                                phi_2216_ = 64f;
                            } else {
                                if _e825_ {
                                    phi_2210_ = true;
                                } else {
                                    phi_2210_ = (_e824_ == 0u);
                                }
                                let _e830_ = phi_2210_;
                                phi_2216_ = bitcast<f32>(select(_e823_, 0u, _e830_));
                            }
                            let _e834_ = phi_2216_;
                            phi_2217_ = _e834_;
                        }
                        let _e836_ = phi_2217_;
                        phi_2218_ = _e836_;
                    }
                    let _e851_ = phi_2218_;
                    if _e158_ {
                        let _e853_ = (_e593_ + _e593_);
                        let _e854_ = (_e594_ + _e594_);
                        phi_2240_ = type_23_(((_e592_ + _e592_) - 1f), -(_e853_), _e853_, (1f - _e854_), (_e853_ * _e194_.x), (_e854_ * _e194_.x));
                    } else {
                        phi_2240_ = type_23_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e863_ = phi_2240_;
                    let _e865_ = select(255u, _e113_, (_e113_ < 255u));
                    let _e867_ = (_e865_ < 1536u);
                    if _e867_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e865_] = _e863_.member;
                    let _e870_ = (256u + _e865_);
                    let _e871_ = (_e870_ < 1536u);
                    if _e871_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e870_] = _e863_.member_1_;
                    let _e874_ = (512u + _e865_);
                    let _e875_ = (_e874_ < 1536u);
                    if _e875_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e874_] = _e863_.member_2_;
                    let _e878_ = (768u + _e865_);
                    let _e879_ = (_e878_ < 1536u);
                    if _e879_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e878_] = _e863_.member_3_;
                    let _e882_ = (1024u + _e865_);
                    let _e883_ = (_e882_ < 1536u);
                    if _e883_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e882_] = _e863_.member_4_;
                    let _e886_ = (1280u + _e865_);
                    let _e887_ = (_e886_ < 1536u);
                    if _e887_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e886_] = _e863_.member_5_;
                    workgroupBarrier();
                    phi_2300_ = 1u;
                    loop {
                        let _e890_ = phi_2300_;
                        let _e891_ = (_e890_ < 256u);
                        if _e891_ {
                            let _e893_ = (_e890_ << bitcast<u32>(1i));
                            let _e896_ = (((_e113_ + 1u) * _e893_) - 1u);
                            if (_e896_ < 256u) {
                                let _e898_ = (_e896_ - _e890_);
                                let _e900_ = select(255u, _e898_, (_e898_ < 255u));
                                if (_e900_ < 1536u) {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e903_ = scan_scratch[_e900_];
                                let _e904_ = (256u + _e900_);
                                if (_e904_ < 1536u) {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e907_ = scan_scratch[_e904_];
                                let _e908_ = (512u + _e900_);
                                if (_e908_ < 1536u) {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e911_ = scan_scratch[_e908_];
                                let _e912_ = (768u + _e900_);
                                if (_e912_ < 1536u) {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e915_ = scan_scratch[_e912_];
                                let _e916_ = (1024u + _e900_);
                                if (_e916_ < 1536u) {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e919_ = scan_scratch[_e916_];
                                let _e920_ = (1280u + _e900_);
                                if (_e920_ < 1536u) {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e923_ = scan_scratch[_e920_];
                                let _e925_ = select(255u, _e896_, (_e896_ < 255u));
                                let _e926_ = (_e925_ < 1536u);
                                if _e926_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e928_ = scan_scratch[_e925_];
                                let _e929_ = (256u + _e925_);
                                let _e930_ = (_e929_ < 1536u);
                                if _e930_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e932_ = scan_scratch[_e929_];
                                let _e933_ = (512u + _e925_);
                                let _e934_ = (_e933_ < 1536u);
                                if _e934_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e936_ = scan_scratch[_e933_];
                                let _e937_ = (768u + _e925_);
                                let _e938_ = (_e937_ < 1536u);
                                if _e938_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e940_ = scan_scratch[_e937_];
                                let _e941_ = (1024u + _e925_);
                                let _e942_ = (_e941_ < 1536u);
                                if _e942_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e944_ = scan_scratch[_e941_];
                                let _e945_ = (1280u + _e925_);
                                let _e946_ = (_e945_ < 1536u);
                                if _e946_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                let _e948_ = scan_scratch[_e945_];
                                if _e926_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                scan_scratch[_e925_] = ((_e928_ * _e903_) + (_e932_ * _e911_));
                                if _e930_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                scan_scratch[_e929_] = ((_e928_ * _e907_) + (_e932_ * _e915_));
                                if _e934_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                scan_scratch[_e933_] = ((_e936_ * _e903_) + (_e940_ * _e911_));
                                if _e938_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                scan_scratch[_e937_] = ((_e936_ * _e907_) + (_e940_ * _e915_));
                                if _e942_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                scan_scratch[_e941_] = (((_e928_ * _e919_) + (_e932_ * _e923_)) + _e944_);
                                if _e946_ {
                                } else {
                                    phi_11808_ = true;
                                    break;
                                }
                                scan_scratch[_e945_] = (((_e936_ * _e919_) + (_e940_ * _e923_)) + _e948_);
                            }
                            workgroupBarrier();
                            phi_2301_ = _e893_;
                        } else {
                            phi_2301_ = u32();
                        }
                        let _e970_ = phi_2301_;
                        phi_2300_ = _e970_;
                        phi_11808_ = _e197_;
                        if !(_e891_) {
                            break;
                        }
                    }
                    let _e973_ = phi_11808_;
                    phi_12444_ = _e973_;
                    if _e973_ {
                        break;
                    }
                    let _e974_ = (_e113_ == 255u);
                    if _e974_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_2563_ = 128u;
                    loop {
                        let _e982_ = phi_2563_;
                        let _e983_ = (_e982_ > 0u);
                        if _e983_ {
                            let _e988_ = (((_e113_ + 1u) * (_e982_ << bitcast<u32>(1i))) - 1u);
                            if (_e988_ < 256u) {
                                let _e990_ = (_e988_ - _e982_);
                                let _e992_ = select(255u, _e990_, (_e990_ < 255u));
                                let _e993_ = (_e992_ < 1536u);
                                if _e993_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e995_ = scan_scratch[_e992_];
                                let _e996_ = (256u + _e992_);
                                let _e997_ = (_e996_ < 1536u);
                                if _e997_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e999_ = scan_scratch[_e996_];
                                let _e1000_ = (512u + _e992_);
                                let _e1001_ = (_e1000_ < 1536u);
                                if _e1001_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1003_ = scan_scratch[_e1000_];
                                let _e1004_ = (768u + _e992_);
                                let _e1005_ = (_e1004_ < 1536u);
                                if _e1005_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1007_ = scan_scratch[_e1004_];
                                let _e1008_ = (1024u + _e992_);
                                let _e1009_ = (_e1008_ < 1536u);
                                if _e1009_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1011_ = scan_scratch[_e1008_];
                                let _e1012_ = (1280u + _e992_);
                                let _e1013_ = (_e1012_ < 1536u);
                                if _e1013_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1015_ = scan_scratch[_e1012_];
                                let _e1017_ = select(255u, _e988_, (_e988_ < 255u));
                                let _e1018_ = (_e1017_ < 1536u);
                                if _e1018_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1020_ = scan_scratch[_e1017_];
                                let _e1021_ = (256u + _e1017_);
                                let _e1022_ = (_e1021_ < 1536u);
                                if _e1022_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1024_ = scan_scratch[_e1021_];
                                let _e1025_ = (512u + _e1017_);
                                let _e1026_ = (_e1025_ < 1536u);
                                if _e1026_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1028_ = scan_scratch[_e1025_];
                                let _e1029_ = (768u + _e1017_);
                                let _e1030_ = (_e1029_ < 1536u);
                                if _e1030_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1032_ = scan_scratch[_e1029_];
                                let _e1033_ = (1024u + _e1017_);
                                let _e1034_ = (_e1033_ < 1536u);
                                if _e1034_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1036_ = scan_scratch[_e1033_];
                                let _e1037_ = (1280u + _e1017_);
                                let _e1038_ = (_e1037_ < 1536u);
                                if _e1038_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                let _e1040_ = scan_scratch[_e1037_];
                                if _e993_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e992_] = _e1020_;
                                if _e997_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e996_] = _e1024_;
                                if _e1001_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1000_] = _e1028_;
                                if _e1005_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1004_] = _e1032_;
                                if _e1009_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1008_] = _e1036_;
                                if _e1013_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1012_] = _e1040_;
                                if _e1018_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1017_] = ((_e995_ * _e1020_) + (_e999_ * _e1028_));
                                if _e1022_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1021_] = ((_e995_ * _e1024_) + (_e999_ * _e1032_));
                                if _e1026_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1025_] = ((_e1003_ * _e1020_) + (_e1007_ * _e1028_));
                                if _e1030_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1029_] = ((_e1003_ * _e1024_) + (_e1007_ * _e1032_));
                                if _e1034_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1033_] = (((_e995_ * _e1036_) + (_e999_ * _e1040_)) + _e1011_);
                                if _e1038_ {
                                } else {
                                    phi_12109_ = true;
                                    break;
                                }
                                scan_scratch[_e1037_] = (((_e1003_ * _e1036_) + (_e1007_ * _e1040_)) + _e1015_);
                            }
                            workgroupBarrier();
                            phi_2564_ = (_e982_ >> bitcast<u32>(1i));
                        } else {
                            phi_2564_ = u32();
                        }
                        let _e1064_ = phi_2564_;
                        phi_2563_ = _e1064_;
                        phi_12109_ = _e973_;
                        if !(_e983_) {
                            break;
                        }
                    }
                    let _e1067_ = phi_12109_;
                    phi_12444_ = _e1067_;
                    if _e1067_ {
                        break;
                    }
                    if _e867_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    if _e871_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    if _e875_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    if _e879_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    if _e883_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e1068_ = scan_scratch[_e882_];
                    if _e887_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e1069_ = scan_scratch[_e886_];
                    workgroupBarrier();
                    if _e158_ {
                        let _e1070_ = (_e194_.x - _e1069_);
                        let _e1073_ = ((_e592_ * _e1068_) + (_e593_ * _e1070_));
                        let _e1077_ = ((_e1069_ + (_e593_ * _e1068_)) + (_e594_ * _e1070_));
                        let _e1088_ = bitcast<u32>(((((_e641_ * _e1077_) + (_e659_ * _e1073_)) + (_e677_ * ((_e194_.x - (_e578_ * _e1073_)) - _e1077_))) + (_e695_ * _e194_.x)));
                        let _e1089_ = (_e1088_ & 2139095040u);
                        if (_e1089_ == 2139095040u) {
                            phi_3106_ = true;
                        } else {
                            phi_3106_ = (_e1089_ == 0u);
                        }
                        let _e1093_ = phi_3106_;
                        let _e1096_ = bitcast<u32>(_e1073_);
                        let _e1097_ = (_e1096_ & 2139095040u);
                        if (_e1097_ == 2139095040u) {
                            phi_3122_ = true;
                        } else {
                            phi_3122_ = (_e1097_ == 0u);
                        }
                        let _e1101_ = phi_3122_;
                        phi_3130_ = type_20_(bitcast<f32>(select(_e1088_, 0u, _e1093_)), bitcast<f32>(select(_e1096_, 0u, _e1101_)));
                    } else {
                        phi_3130_ = type_20_(0f, 0f);
                    }
                    let _e1106_ = phi_3130_;
                    if _e158_ {
                        let _e1110_ = (_e593_ + _e593_);
                        let _e1111_ = (_e594_ + _e594_);
                        phi_3153_ = type_23_(((_e592_ + _e592_) - 1f), -(_e1110_), _e1110_, (1f - _e1111_), (_e1110_ * _e194_.y), (_e1111_ * _e194_.y));
                    } else {
                        phi_3153_ = type_23_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e1120_ = phi_3153_;
                    if _e867_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e865_] = _e1120_.member;
                    if _e871_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e870_] = _e1120_.member_1_;
                    if _e875_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e874_] = _e1120_.member_2_;
                    if _e879_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e878_] = _e1120_.member_3_;
                    if _e883_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e882_] = _e1120_.member_4_;
                    if _e887_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e886_] = _e1120_.member_5_;
                    workgroupBarrier();
                    phi_3213_ = 1u;
                    loop {
                        let _e1128_ = phi_3213_;
                        let _e1129_ = (_e1128_ < 256u);
                        if _e1129_ {
                            let _e1131_ = (_e1128_ << bitcast<u32>(1i));
                            let _e1134_ = (((_e113_ + 1u) * _e1131_) - 1u);
                            if (_e1134_ < 256u) {
                                let _e1136_ = (_e1134_ - _e1128_);
                                let _e1138_ = select(255u, _e1136_, (_e1136_ < 255u));
                                if (_e1138_ < 1536u) {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1141_ = scan_scratch[_e1138_];
                                let _e1142_ = (256u + _e1138_);
                                if (_e1142_ < 1536u) {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1145_ = scan_scratch[_e1142_];
                                let _e1146_ = (512u + _e1138_);
                                if (_e1146_ < 1536u) {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1149_ = scan_scratch[_e1146_];
                                let _e1150_ = (768u + _e1138_);
                                if (_e1150_ < 1536u) {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1153_ = scan_scratch[_e1150_];
                                let _e1154_ = (1024u + _e1138_);
                                if (_e1154_ < 1536u) {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1157_ = scan_scratch[_e1154_];
                                let _e1158_ = (1280u + _e1138_);
                                if (_e1158_ < 1536u) {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1161_ = scan_scratch[_e1158_];
                                let _e1163_ = select(255u, _e1134_, (_e1134_ < 255u));
                                let _e1164_ = (_e1163_ < 1536u);
                                if _e1164_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1166_ = scan_scratch[_e1163_];
                                let _e1167_ = (256u + _e1163_);
                                let _e1168_ = (_e1167_ < 1536u);
                                if _e1168_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1170_ = scan_scratch[_e1167_];
                                let _e1171_ = (512u + _e1163_);
                                let _e1172_ = (_e1171_ < 1536u);
                                if _e1172_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1174_ = scan_scratch[_e1171_];
                                let _e1175_ = (768u + _e1163_);
                                let _e1176_ = (_e1175_ < 1536u);
                                if _e1176_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1178_ = scan_scratch[_e1175_];
                                let _e1179_ = (1024u + _e1163_);
                                let _e1180_ = (_e1179_ < 1536u);
                                if _e1180_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1182_ = scan_scratch[_e1179_];
                                let _e1183_ = (1280u + _e1163_);
                                let _e1184_ = (_e1183_ < 1536u);
                                if _e1184_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                let _e1186_ = scan_scratch[_e1183_];
                                if _e1164_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                scan_scratch[_e1163_] = ((_e1166_ * _e1141_) + (_e1170_ * _e1149_));
                                if _e1168_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                scan_scratch[_e1167_] = ((_e1166_ * _e1145_) + (_e1170_ * _e1153_));
                                if _e1172_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                scan_scratch[_e1171_] = ((_e1174_ * _e1141_) + (_e1178_ * _e1149_));
                                if _e1176_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                scan_scratch[_e1175_] = ((_e1174_ * _e1145_) + (_e1178_ * _e1153_));
                                if _e1180_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                scan_scratch[_e1179_] = (((_e1166_ * _e1157_) + (_e1170_ * _e1161_)) + _e1182_);
                                if _e1184_ {
                                } else {
                                    phi_12124_ = true;
                                    break;
                                }
                                scan_scratch[_e1183_] = (((_e1174_ * _e1157_) + (_e1178_ * _e1161_)) + _e1186_);
                            }
                            workgroupBarrier();
                            phi_3214_ = _e1131_;
                        } else {
                            phi_3214_ = u32();
                        }
                        let _e1208_ = phi_3214_;
                        phi_3213_ = _e1208_;
                        phi_12124_ = _e1067_;
                        if !(_e1129_) {
                            break;
                        }
                    }
                    let _e1211_ = phi_12124_;
                    phi_12444_ = _e1211_;
                    if _e1211_ {
                        break;
                    }
                    if _e974_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_3476_ = 128u;
                    loop {
                        let _e1219_ = phi_3476_;
                        let _e1220_ = (_e1219_ > 0u);
                        if _e1220_ {
                            let _e1225_ = (((_e113_ + 1u) * (_e1219_ << bitcast<u32>(1i))) - 1u);
                            if (_e1225_ < 256u) {
                                let _e1227_ = (_e1225_ - _e1219_);
                                let _e1229_ = select(255u, _e1227_, (_e1227_ < 255u));
                                let _e1230_ = (_e1229_ < 1536u);
                                if _e1230_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1232_ = scan_scratch[_e1229_];
                                let _e1233_ = (256u + _e1229_);
                                let _e1234_ = (_e1233_ < 1536u);
                                if _e1234_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1236_ = scan_scratch[_e1233_];
                                let _e1237_ = (512u + _e1229_);
                                let _e1238_ = (_e1237_ < 1536u);
                                if _e1238_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1240_ = scan_scratch[_e1237_];
                                let _e1241_ = (768u + _e1229_);
                                let _e1242_ = (_e1241_ < 1536u);
                                if _e1242_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1244_ = scan_scratch[_e1241_];
                                let _e1245_ = (1024u + _e1229_);
                                let _e1246_ = (_e1245_ < 1536u);
                                if _e1246_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1248_ = scan_scratch[_e1245_];
                                let _e1249_ = (1280u + _e1229_);
                                let _e1250_ = (_e1249_ < 1536u);
                                if _e1250_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1252_ = scan_scratch[_e1249_];
                                let _e1254_ = select(255u, _e1225_, (_e1225_ < 255u));
                                let _e1255_ = (_e1254_ < 1536u);
                                if _e1255_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1257_ = scan_scratch[_e1254_];
                                let _e1258_ = (256u + _e1254_);
                                let _e1259_ = (_e1258_ < 1536u);
                                if _e1259_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1261_ = scan_scratch[_e1258_];
                                let _e1262_ = (512u + _e1254_);
                                let _e1263_ = (_e1262_ < 1536u);
                                if _e1263_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1265_ = scan_scratch[_e1262_];
                                let _e1266_ = (768u + _e1254_);
                                let _e1267_ = (_e1266_ < 1536u);
                                if _e1267_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1269_ = scan_scratch[_e1266_];
                                let _e1270_ = (1024u + _e1254_);
                                let _e1271_ = (_e1270_ < 1536u);
                                if _e1271_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1273_ = scan_scratch[_e1270_];
                                let _e1274_ = (1280u + _e1254_);
                                let _e1275_ = (_e1274_ < 1536u);
                                if _e1275_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                let _e1277_ = scan_scratch[_e1274_];
                                if _e1230_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1229_] = _e1257_;
                                if _e1234_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1233_] = _e1261_;
                                if _e1238_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1237_] = _e1265_;
                                if _e1242_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1241_] = _e1269_;
                                if _e1246_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1245_] = _e1273_;
                                if _e1250_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1249_] = _e1277_;
                                if _e1255_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1254_] = ((_e1232_ * _e1257_) + (_e1236_ * _e1265_));
                                if _e1259_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1258_] = ((_e1232_ * _e1261_) + (_e1236_ * _e1269_));
                                if _e1263_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1262_] = ((_e1240_ * _e1257_) + (_e1244_ * _e1265_));
                                if _e1267_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1266_] = ((_e1240_ * _e1261_) + (_e1244_ * _e1269_));
                                if _e1271_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1270_] = (((_e1232_ * _e1273_) + (_e1236_ * _e1277_)) + _e1248_);
                                if _e1275_ {
                                } else {
                                    phi_12144_ = true;
                                    break;
                                }
                                scan_scratch[_e1274_] = (((_e1240_ * _e1273_) + (_e1244_ * _e1277_)) + _e1252_);
                            }
                            workgroupBarrier();
                            phi_3477_ = (_e1219_ >> bitcast<u32>(1i));
                        } else {
                            phi_3477_ = u32();
                        }
                        let _e1301_ = phi_3477_;
                        phi_3476_ = _e1301_;
                        phi_12144_ = _e1211_;
                        if !(_e1220_) {
                            break;
                        }
                    }
                    let _e1304_ = phi_12144_;
                    phi_12444_ = _e1304_;
                    if _e1304_ {
                        break;
                    }
                    if _e867_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    if _e871_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    if _e875_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    if _e879_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    if _e883_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e1305_ = scan_scratch[_e882_];
                    if _e887_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    let _e1306_ = scan_scratch[_e886_];
                    workgroupBarrier();
                    if _e158_ {
                        let _e1307_ = (_e194_.y - _e1306_);
                        let _e1310_ = ((_e592_ * _e1305_) + (_e593_ * _e1307_));
                        let _e1314_ = ((_e1306_ + (_e593_ * _e1305_)) + (_e594_ * _e1307_));
                        let _e1325_ = bitcast<u32>(((((_e641_ * _e1314_) + (_e659_ * _e1310_)) + (_e677_ * ((_e194_.y - (_e578_ * _e1310_)) - _e1314_))) + (_e695_ * _e194_.y)));
                        let _e1326_ = (_e1325_ & 2139095040u);
                        if (_e1326_ == 2139095040u) {
                            phi_4019_ = true;
                        } else {
                            phi_4019_ = (_e1326_ == 0u);
                        }
                        let _e1330_ = phi_4019_;
                        let _e1333_ = bitcast<u32>(_e1310_);
                        let _e1334_ = (_e1333_ & 2139095040u);
                        if (_e1334_ == 2139095040u) {
                            phi_4035_ = true;
                        } else {
                            phi_4035_ = (_e1334_ == 0u);
                        }
                        let _e1338_ = phi_4035_;
                        phi_4043_ = type_20_(bitcast<f32>(select(_e1325_, 0u, _e1330_)), bitcast<f32>(select(_e1333_, 0u, _e1338_)));
                    } else {
                        phi_4043_ = type_20_(0f, 0f);
                    }
                    let _e1343_ = phi_4043_;
                    if _e867_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e865_] = _e1106_.member;
                    if _e871_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e870_] = _e1343_.member;
                    if _e875_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e874_] = _e1106_.member_1_;
                    if _e879_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e878_] = _e1343_.member_1_;
                    if _e883_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e882_] = 0f;
                    if _e887_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e886_] = 0f;
                    workgroupBarrier();
                    if (_e113_ == 0u) {
                        phi_4108_ = type_15_(0u, select(_e154_, 256u, (256u < _e154_)));
                        phi_4111_ = type_22_(array<f32, 4>(0f, 0f, 0f, 0f));
                        phi_4113_ = type_22_(array<f32, 4>(0f, 0f, 0f, 0f));
                        loop {
                            let _e1349_ = phi_4108_;
                            let _e1351_ = phi_4111_;
                            let _e1353_ = phi_4113_;
                            if (_e1349_.member < _e1349_.member_1_) {
                                phi_4109_ = type_15_((_e1349_.member + 1u), _e1349_.member_1_);
                                phi_4136_ = type_15_(1u, _e1349_.member);
                            } else {
                                phi_4109_ = _e1349_;
                                phi_4136_ = type_15_(0u, u32());
                            }
                            let _e1366_ = phi_4109_;
                            let _e1368_ = phi_4136_;
                            let _e1372_ = (bitcast<i32>(_e1368_.member) != 0i);
                            if _e1372_ {
                                if (_e1368_.member_1_ < 1536u) {
                                } else {
                                    phi_12159_ = true;
                                    break;
                                }
                                let _e1375_ = scan_scratch[_e1368_.member_1_];
                                let _e1376_ = (256u + _e1368_.member_1_);
                                if (_e1376_ < 1536u) {
                                } else {
                                    phi_12159_ = true;
                                    break;
                                }
                                let _e1379_ = scan_scratch[_e1376_];
                                let _e1380_ = (512u + _e1368_.member_1_);
                                if (_e1380_ < 1536u) {
                                } else {
                                    phi_12159_ = true;
                                    break;
                                }
                                let _e1383_ = scan_scratch[_e1380_];
                                let _e1384_ = (768u + _e1368_.member_1_);
                                if (_e1384_ < 1536u) {
                                } else {
                                    phi_12159_ = true;
                                    break;
                                }
                                let _e1387_ = scan_scratch[_e1384_];
                                let _e1388_ = (_e734_ * _e1375_);
                                let _e1389_ = (_e734_ * _e1379_);
                                let _e1392_ = bitcast<u32>((_e623_ * _e1383_));
                                let _e1393_ = (_e1392_ & 2139095040u);
                                let _e1394_ = (_e1393_ == 2139095040u);
                                if _e1394_ {
                                    phi_4189_ = true;
                                } else {
                                    phi_4189_ = (_e1393_ == 0u);
                                }
                                let _e1397_ = phi_4189_;
                                if _e1394_ {
                                    phi_4205_ = true;
                                } else {
                                    phi_4205_ = (_e1393_ == 0u);
                                }
                                let _e1402_ = phi_4205_;
                                if _e1394_ {
                                    phi_4221_ = true;
                                } else {
                                    phi_4221_ = (_e1393_ == 0u);
                                }
                                let _e1407_ = phi_4221_;
                                if _e1394_ {
                                    phi_4237_ = true;
                                } else {
                                    phi_4237_ = (_e1393_ == 0u);
                                }
                                let _e1412_ = phi_4237_;
                                let _e1431_ = (_e556_ <= 0f);
                                let _e1436_ = (1f - select(select(select(0f, 1f, (1f <= _e460_)), _e460_, (_e460_ <= 1f)), 0f, (_e460_ <= 0f)));
                                let _e1437_ = (_e1436_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1392_, 0u, _e1397_))) & 2147483647u))), 0f, _e1431_));
                                let _e1438_ = (_e1436_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1392_, 0u, _e1402_))) & 2147483647u))), 0f, _e1431_));
                                let _e1439_ = (_e1436_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1392_, 0u, _e1407_))) & 2147483647u))), 0f, _e1431_));
                                let _e1440_ = (_e1436_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1392_, 0u, _e1412_))) & 2147483647u))), 0f, _e1431_));
                                if (_e1437_ != _e1437_) {
                                    phi_8428_ = true;
                                } else {
                                    phi_8428_ = (0f >= _e1437_);
                                }
                                let _e1444_ = phi_8428_;
                                let _e1445_ = select(_e1437_, 0f, _e1444_);
                                if (_e1438_ != _e1438_) {
                                    phi_8443_ = true;
                                } else {
                                    phi_8443_ = (0f >= _e1438_);
                                }
                                let _e1449_ = phi_8443_;
                                let _e1450_ = select(_e1438_, 0f, _e1449_);
                                if (_e1439_ != _e1439_) {
                                    phi_8458_ = true;
                                } else {
                                    phi_8458_ = (0f >= _e1439_);
                                }
                                let _e1454_ = phi_8458_;
                                let _e1455_ = select(_e1439_, 0f, _e1454_);
                                if (_e1440_ != _e1440_) {
                                    phi_8473_ = true;
                                } else {
                                    phi_8473_ = (0f >= _e1440_);
                                }
                                let _e1459_ = phi_8473_;
                                let _e1460_ = select(_e1440_, 0f, _e1459_);
                                if (_e1445_ != _e1445_) {
                                    phi_8488_ = true;
                                } else {
                                    phi_8488_ = (1f <= _e1445_);
                                }
                                let _e1464_ = phi_8488_;
                                let _e1465_ = select(_e1445_, 1f, _e1464_);
                                if (_e1450_ != _e1450_) {
                                    phi_8503_ = true;
                                } else {
                                    phi_8503_ = (1f <= _e1450_);
                                }
                                let _e1469_ = phi_8503_;
                                let _e1470_ = select(_e1450_, 1f, _e1469_);
                                if (_e1455_ != _e1455_) {
                                    phi_8518_ = true;
                                } else {
                                    phi_8518_ = (1f <= _e1455_);
                                }
                                let _e1474_ = phi_8518_;
                                let _e1475_ = select(_e1455_, 1f, _e1474_);
                                if (_e1460_ != _e1460_) {
                                    phi_8533_ = true;
                                } else {
                                    phi_8533_ = (1f <= _e1460_);
                                }
                                let _e1479_ = phi_8533_;
                                let _e1480_ = select(_e1460_, 1f, _e1479_);
                                let _e1483_ = (_e538_ * _e1353_.member[0]);
                                let _e1486_ = (_e538_ * _e1353_.member[1]);
                                let _e1489_ = (_e538_ * _e1353_.member[2]);
                                let _e1492_ = (_e538_ * _e1353_.member[3]);
                                let _e1493_ = (1f - _e562_);
                                let _e1494_ = (select(select(select(0f, 1f, (1f <= _e478_)), _e478_, (_e478_ <= 1f)), 0f, (_e478_ <= 0f)) * _e1493_);
                                let _e1496_ = (_e1388_ - (_e1494_ * _e1483_));
                                let _e1498_ = (_e1388_ - (_e1494_ * _e1486_));
                                let _e1500_ = (_e1388_ - (_e1494_ * _e1489_));
                                let _e1502_ = (_e1388_ - (_e1494_ * _e1492_));
                                if (_e1496_ != _e1496_) {
                                    phi_8548_ = true;
                                } else {
                                    phi_8548_ = (-1f >= _e1496_);
                                }
                                let _e1506_ = phi_8548_;
                                let _e1507_ = select(_e1496_, -1f, _e1506_);
                                if (_e1498_ != _e1498_) {
                                    phi_8563_ = true;
                                } else {
                                    phi_8563_ = (-1f >= _e1498_);
                                }
                                let _e1511_ = phi_8563_;
                                let _e1512_ = select(_e1498_, -1f, _e1511_);
                                if (_e1500_ != _e1500_) {
                                    phi_8578_ = true;
                                } else {
                                    phi_8578_ = (-1f >= _e1500_);
                                }
                                let _e1516_ = phi_8578_;
                                let _e1517_ = select(_e1500_, -1f, _e1516_);
                                if (_e1502_ != _e1502_) {
                                    phi_8593_ = true;
                                } else {
                                    phi_8593_ = (-1f >= _e1502_);
                                }
                                let _e1521_ = phi_8593_;
                                let _e1522_ = select(_e1502_, -1f, _e1521_);
                                if (_e1507_ != _e1507_) {
                                    phi_8608_ = true;
                                } else {
                                    phi_8608_ = (1f <= _e1507_);
                                }
                                let _e1526_ = phi_8608_;
                                let _e1527_ = select(_e1507_, 1f, _e1526_);
                                if (_e1512_ != _e1512_) {
                                    phi_8623_ = true;
                                } else {
                                    phi_8623_ = (1f <= _e1512_);
                                }
                                let _e1531_ = phi_8623_;
                                let _e1532_ = select(_e1512_, 1f, _e1531_);
                                if (_e1517_ != _e1517_) {
                                    phi_8638_ = true;
                                } else {
                                    phi_8638_ = (1f <= _e1517_);
                                }
                                let _e1536_ = phi_8638_;
                                let _e1537_ = select(_e1517_, 1f, _e1536_);
                                if (_e1522_ != _e1522_) {
                                    phi_8653_ = true;
                                } else {
                                    phi_8653_ = (1f <= _e1522_);
                                }
                                let _e1541_ = phi_8653_;
                                let _e1542_ = select(_e1522_, 1f, _e1541_);
                                let _e1561_ = bitcast<f32>((bitcast<u32>(_e1496_) & 2147483647u));
                                let _e1564_ = bitcast<f32>((bitcast<u32>(_e1498_) & 2147483647u));
                                let _e1567_ = bitcast<f32>((bitcast<u32>(_e1500_) & 2147483647u));
                                let _e1570_ = bitcast<f32>((bitcast<u32>(_e1502_) & 2147483647u));
                                if (_e1561_ != _e1561_) {
                                    phi_8668_ = true;
                                } else {
                                    phi_8668_ = (1000000000000000f <= _e1561_);
                                }
                                let _e1574_ = phi_8668_;
                                let _e1575_ = select(_e1561_, 1000000000000000f, _e1574_);
                                if (_e1564_ != _e1564_) {
                                    phi_8683_ = true;
                                } else {
                                    phi_8683_ = (1000000000000000f <= _e1564_);
                                }
                                let _e1579_ = phi_8683_;
                                let _e1580_ = select(_e1564_, 1000000000000000f, _e1579_);
                                if (_e1567_ != _e1567_) {
                                    phi_8698_ = true;
                                } else {
                                    phi_8698_ = (1000000000000000f <= _e1567_);
                                }
                                let _e1584_ = phi_8698_;
                                let _e1585_ = select(_e1567_, 1000000000000000f, _e1584_);
                                if (_e1570_ != _e1570_) {
                                    phi_8713_ = true;
                                } else {
                                    phi_8713_ = (1000000000000000f <= _e1570_);
                                }
                                let _e1589_ = phi_8713_;
                                let _e1590_ = select(_e1570_, 1000000000000000f, _e1589_);
                                let _e1591_ = (_e1575_ * _e1575_);
                                let _e1592_ = (_e1580_ * _e1580_);
                                let _e1593_ = (_e1585_ * _e1585_);
                                let _e1594_ = (_e1590_ * _e1590_);
                                let _e1595_ = (_e1591_ + 0.0000010000001f);
                                let _e1596_ = (_e1592_ + 0.0000010000001f);
                                let _e1597_ = (_e1593_ + 0.0000010000001f);
                                let _e1598_ = (_e1594_ + 0.0000010000001f);
                                if (_e1595_ != _e1595_) {
                                    phi_8728_ = true;
                                } else {
                                    phi_8728_ = (0.000000000001f >= _e1595_);
                                }
                                let _e1602_ = phi_8728_;
                                let _e1603_ = select(_e1595_, 0.000000000001f, _e1602_);
                                if (_e1596_ != _e1596_) {
                                    phi_8743_ = true;
                                } else {
                                    phi_8743_ = (0.000000000001f >= _e1596_);
                                }
                                let _e1607_ = phi_8743_;
                                let _e1608_ = select(_e1596_, 0.000000000001f, _e1607_);
                                if (_e1597_ != _e1597_) {
                                    phi_8758_ = true;
                                } else {
                                    phi_8758_ = (0.000000000001f >= _e1597_);
                                }
                                let _e1612_ = phi_8758_;
                                let _e1613_ = select(_e1597_, 0.000000000001f, _e1612_);
                                if (_e1598_ != _e1598_) {
                                    phi_8773_ = true;
                                } else {
                                    phi_8773_ = (0.000000000001f >= _e1598_);
                                }
                                let _e1617_ = phi_8773_;
                                let _e1618_ = select(_e1598_, 0.000000000001f, _e1617_);
                                let _e1619_ = (_e1603_ * 0.5f);
                                let _e1620_ = (_e1608_ * 0.5f);
                                let _e1621_ = (_e1613_ * 0.5f);
                                let _e1622_ = (_e1618_ * 0.5f);
                                let _e1627_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1603_) >> bitcast<u32>(1i))));
                                let _e1632_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1608_) >> bitcast<u32>(1i))));
                                let _e1637_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1613_) >> bitcast<u32>(1i))));
                                let _e1642_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1618_) >> bitcast<u32>(1i))));
                                let _e1655_ = (_e1627_ * (1.5f - (_e1619_ * (_e1627_ * _e1627_))));
                                let _e1656_ = (_e1632_ * (1.5f - (_e1620_ * (_e1632_ * _e1632_))));
                                let _e1657_ = (_e1637_ * (1.5f - (_e1621_ * (_e1637_ * _e1637_))));
                                let _e1658_ = (_e1642_ * (1.5f - (_e1622_ * (_e1642_ * _e1642_))));
                                let _e1671_ = (_e1655_ * (1.5f - (_e1619_ * (_e1655_ * _e1655_))));
                                let _e1672_ = (_e1656_ * (1.5f - (_e1620_ * (_e1656_ * _e1656_))));
                                let _e1673_ = (_e1657_ * (1.5f - (_e1621_ * (_e1657_ * _e1657_))));
                                let _e1674_ = (_e1658_ * (1.5f - (_e1622_ * (_e1658_ * _e1658_))));
                                let _e1691_ = (_e1595_ * (_e1671_ * (1.5f - (_e1619_ * (_e1671_ * _e1671_)))));
                                let _e1692_ = (_e1596_ * (_e1672_ * (1.5f - (_e1620_ * (_e1672_ * _e1672_)))));
                                let _e1693_ = (_e1597_ * (_e1673_ * (1.5f - (_e1621_ * (_e1673_ * _e1673_)))));
                                let _e1694_ = (_e1598_ * (_e1674_ * (1.5f - (_e1622_ * (_e1674_ * _e1674_)))));
                                let _e1695_ = (_e1691_ + 0.001f);
                                let _e1696_ = (_e1692_ + 0.001f);
                                let _e1697_ = (_e1693_ + 0.001f);
                                let _e1698_ = (_e1694_ + 0.001f);
                                if (_e1695_ != _e1695_) {
                                    phi_8788_ = true;
                                } else {
                                    phi_8788_ = (0.000001f >= _e1695_);
                                }
                                let _e1702_ = phi_8788_;
                                let _e1703_ = select(_e1695_, 0.000001f, _e1702_);
                                if (_e1696_ != _e1696_) {
                                    phi_8803_ = true;
                                } else {
                                    phi_8803_ = (0.000001f >= _e1696_);
                                }
                                let _e1707_ = phi_8803_;
                                let _e1708_ = select(_e1696_, 0.000001f, _e1707_);
                                if (_e1697_ != _e1697_) {
                                    phi_8818_ = true;
                                } else {
                                    phi_8818_ = (0.000001f >= _e1697_);
                                }
                                let _e1712_ = phi_8818_;
                                let _e1713_ = select(_e1697_, 0.000001f, _e1712_);
                                if (_e1698_ != _e1698_) {
                                    phi_8833_ = true;
                                } else {
                                    phi_8833_ = (0.000001f >= _e1698_);
                                }
                                let _e1717_ = phi_8833_;
                                let _e1718_ = select(_e1698_, 0.000001f, _e1717_);
                                let _e1721_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1703_)));
                                let _e1724_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1708_)));
                                let _e1727_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1713_)));
                                let _e1730_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1718_)));
                                let _e1740_ = (_e1721_ + (_e1721_ * (1f - (_e1703_ * _e1721_))));
                                let _e1742_ = (_e1724_ + (_e1724_ * (1f - (_e1708_ * _e1724_))));
                                let _e1744_ = (_e1727_ + (_e1727_ * (1f - (_e1713_ * _e1727_))));
                                let _e1746_ = (_e1730_ + (_e1730_ * (1f - (_e1718_ * _e1730_))));
                                let _e1756_ = (_e1740_ + (_e1740_ * (1f - (_e1703_ * _e1740_))));
                                let _e1758_ = (_e1742_ + (_e1742_ * (1f - (_e1708_ * _e1742_))));
                                let _e1760_ = (_e1744_ + (_e1744_ * (1f - (_e1713_ * _e1744_))));
                                let _e1762_ = (_e1746_ + (_e1746_ * (1f - (_e1718_ * _e1746_))));
                                let _e1772_ = (_e1756_ + (_e1756_ * (1f - (_e1703_ * _e1756_))));
                                let _e1774_ = (_e1758_ + (_e1758_ * (1f - (_e1708_ * _e1758_))));
                                let _e1776_ = (_e1760_ + (_e1760_ * (1f - (_e1713_ * _e1760_))));
                                let _e1778_ = (_e1762_ + (_e1762_ * (1f - (_e1718_ * _e1762_))));
                                let _e1779_ = (_e1691_ + _e1575_);
                                let _e1780_ = (_e1692_ + _e1580_);
                                let _e1781_ = (_e1693_ + _e1585_);
                                let _e1782_ = (_e1694_ + _e1590_);
                                if (_e1779_ != _e1779_) {
                                    phi_8848_ = true;
                                } else {
                                    phi_8848_ = (0.000001f >= _e1779_);
                                }
                                let _e1786_ = phi_8848_;
                                let _e1787_ = select(_e1779_, 0.000001f, _e1786_);
                                if (_e1780_ != _e1780_) {
                                    phi_8863_ = true;
                                } else {
                                    phi_8863_ = (0.000001f >= _e1780_);
                                }
                                let _e1791_ = phi_8863_;
                                let _e1792_ = select(_e1780_, 0.000001f, _e1791_);
                                if (_e1781_ != _e1781_) {
                                    phi_8878_ = true;
                                } else {
                                    phi_8878_ = (0.000001f >= _e1781_);
                                }
                                let _e1796_ = phi_8878_;
                                let _e1797_ = select(_e1781_, 0.000001f, _e1796_);
                                if (_e1782_ != _e1782_) {
                                    phi_8893_ = true;
                                } else {
                                    phi_8893_ = (0.000001f >= _e1782_);
                                }
                                let _e1801_ = phi_8893_;
                                let _e1802_ = select(_e1782_, 0.000001f, _e1801_);
                                let _e1805_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1787_)));
                                let _e1808_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1792_)));
                                let _e1811_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1797_)));
                                let _e1814_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1802_)));
                                let _e1824_ = (_e1805_ + (_e1805_ * (1f - (_e1787_ * _e1805_))));
                                let _e1826_ = (_e1808_ + (_e1808_ * (1f - (_e1792_ * _e1808_))));
                                let _e1828_ = (_e1811_ + (_e1811_ * (1f - (_e1797_ * _e1811_))));
                                let _e1830_ = (_e1814_ + (_e1814_ * (1f - (_e1802_ * _e1814_))));
                                let _e1840_ = (_e1824_ + (_e1824_ * (1f - (_e1787_ * _e1824_))));
                                let _e1842_ = (_e1826_ + (_e1826_ * (1f - (_e1792_ * _e1826_))));
                                let _e1844_ = (_e1828_ + (_e1828_ * (1f - (_e1797_ * _e1828_))));
                                let _e1846_ = (_e1830_ + (_e1830_ * (1f - (_e1802_ * _e1830_))));
                                let _e1883_ = (_e1561_ <= 1000000000000000f);
                                let _e1885_ = (_e1564_ <= 1000000000000000f);
                                let _e1887_ = (_e1567_ <= 1000000000000000f);
                                let _e1889_ = (_e1570_ <= 1000000000000000f);
                                let _e1920_ = select(select(select(_e1561_, (0.5f * (_e1575_ + (_e1591_ * _e1772_))), _e1883_), select(-0.0005f, -(((0.0005f * (_e1575_ * (_e1840_ + (_e1840_ * (1f - (_e1787_ * _e1840_)))))) * (1f + (_e1575_ * _e1772_)))), _e1883_), (_e1496_ <= 0f)), 0f, (_e1561_ <= 0f));
                                let _e1922_ = select(select(select(_e1564_, (0.5f * (_e1580_ + (_e1592_ * _e1774_))), _e1885_), select(-0.0005f, -(((0.0005f * (_e1580_ * (_e1842_ + (_e1842_ * (1f - (_e1792_ * _e1842_)))))) * (1f + (_e1580_ * _e1774_)))), _e1885_), (_e1498_ <= 0f)), 0f, (_e1564_ <= 0f));
                                let _e1924_ = select(select(select(_e1567_, (0.5f * (_e1585_ + (_e1593_ * _e1776_))), _e1887_), select(-0.0005f, -(((0.0005f * (_e1585_ * (_e1844_ + (_e1844_ * (1f - (_e1797_ * _e1844_)))))) * (1f + (_e1585_ * _e1776_)))), _e1887_), (_e1500_ <= 0f)), 0f, (_e1567_ <= 0f));
                                let _e1926_ = select(select(select(_e1570_, (0.5f * (_e1590_ + (_e1594_ * _e1778_))), _e1889_), select(-0.0005f, -(((0.0005f * (_e1590_ * (_e1846_ + (_e1846_ * (1f - (_e1802_ * _e1846_)))))) * (1f + (_e1590_ * _e1778_)))), _e1889_), (_e1502_ <= 0f)), 0f, (_e1570_ <= 0f));
                                let _e1939_ = (_e562_ <= 0f);
                                let _e1964_ = bitcast<u32>((((1f - (_e562_ * _e1465_)) * _e1483_) + (_e1465_ * select(((_e1493_ * _e1920_) + (_e562_ * (_e1527_ * (1f - (0.33333334f * (_e1527_ * _e1527_)))))), _e1920_, _e1939_))));
                                let _e1965_ = (_e1964_ & 2139095040u);
                                if (_e1965_ == 2139095040u) {
                                    phi_4945_ = true;
                                } else {
                                    phi_4945_ = (_e1965_ == 0u);
                                }
                                let _e1969_ = phi_4945_;
                                let _e1971_ = bitcast<f32>(select(_e1964_, 0u, _e1969_));
                                let _e1972_ = bitcast<u32>((((1f - (_e562_ * _e1470_)) * _e1486_) + (_e1470_ * select(((_e1493_ * _e1922_) + (_e562_ * (_e1532_ * (1f - (0.33333334f * (_e1532_ * _e1532_)))))), _e1922_, _e1939_))));
                                let _e1973_ = (_e1972_ & 2139095040u);
                                if (_e1973_ == 2139095040u) {
                                    phi_4961_ = true;
                                } else {
                                    phi_4961_ = (_e1973_ == 0u);
                                }
                                let _e1977_ = phi_4961_;
                                let _e1980_ = bitcast<u32>((((1f - (_e562_ * _e1475_)) * _e1489_) + (_e1475_ * select(((_e1493_ * _e1924_) + (_e562_ * (_e1537_ * (1f - (0.33333334f * (_e1537_ * _e1537_)))))), _e1924_, _e1939_))));
                                let _e1981_ = (_e1980_ & 2139095040u);
                                if (_e1981_ == 2139095040u) {
                                    phi_4977_ = true;
                                } else {
                                    phi_4977_ = (_e1981_ == 0u);
                                }
                                let _e1985_ = phi_4977_;
                                let _e1988_ = bitcast<u32>((((1f - (_e562_ * _e1480_)) * _e1492_) + (_e1480_ * select(((_e1493_ * _e1926_) + (_e562_ * (_e1542_ * (1f - (0.33333334f * (_e1542_ * _e1542_)))))), _e1926_, _e1939_))));
                                let _e1989_ = (_e1988_ & 2139095040u);
                                if (_e1989_ == 2139095040u) {
                                    phi_4993_ = true;
                                } else {
                                    phi_4993_ = (_e1989_ == 0u);
                                }
                                let _e1993_ = phi_4993_;
                                let _e1998_ = bitcast<u32>((_e623_ * _e1387_));
                                let _e1999_ = (_e1998_ & 2139095040u);
                                let _e2000_ = (_e1999_ == 2139095040u);
                                if _e2000_ {
                                    phi_5015_ = true;
                                } else {
                                    phi_5015_ = (_e1999_ == 0u);
                                }
                                let _e2003_ = phi_5015_;
                                if _e2000_ {
                                    phi_5031_ = true;
                                } else {
                                    phi_5031_ = (_e1999_ == 0u);
                                }
                                let _e2008_ = phi_5031_;
                                if _e2000_ {
                                    phi_5047_ = true;
                                } else {
                                    phi_5047_ = (_e1999_ == 0u);
                                }
                                let _e2013_ = phi_5047_;
                                if _e2000_ {
                                    phi_5063_ = true;
                                } else {
                                    phi_5063_ = (_e1999_ == 0u);
                                }
                                let _e2018_ = phi_5063_;
                                let _e2041_ = (_e1436_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1998_, 0u, _e2003_))) & 2147483647u))), 0f, _e1431_));
                                let _e2042_ = (_e1436_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1998_, 0u, _e2008_))) & 2147483647u))), 0f, _e1431_));
                                let _e2043_ = (_e1436_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1998_, 0u, _e2013_))) & 2147483647u))), 0f, _e1431_));
                                let _e2044_ = (_e1436_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1998_, 0u, _e2018_))) & 2147483647u))), 0f, _e1431_));
                                if (_e2041_ != _e2041_) {
                                    phi_9028_ = true;
                                } else {
                                    phi_9028_ = (0f >= _e2041_);
                                }
                                let _e2048_ = phi_9028_;
                                let _e2049_ = select(_e2041_, 0f, _e2048_);
                                if (_e2042_ != _e2042_) {
                                    phi_9043_ = true;
                                } else {
                                    phi_9043_ = (0f >= _e2042_);
                                }
                                let _e2053_ = phi_9043_;
                                let _e2054_ = select(_e2042_, 0f, _e2053_);
                                if (_e2043_ != _e2043_) {
                                    phi_9058_ = true;
                                } else {
                                    phi_9058_ = (0f >= _e2043_);
                                }
                                let _e2058_ = phi_9058_;
                                let _e2059_ = select(_e2043_, 0f, _e2058_);
                                if (_e2044_ != _e2044_) {
                                    phi_9073_ = true;
                                } else {
                                    phi_9073_ = (0f >= _e2044_);
                                }
                                let _e2063_ = phi_9073_;
                                let _e2064_ = select(_e2044_, 0f, _e2063_);
                                if (_e2049_ != _e2049_) {
                                    phi_9088_ = true;
                                } else {
                                    phi_9088_ = (1f <= _e2049_);
                                }
                                let _e2068_ = phi_9088_;
                                let _e2069_ = select(_e2049_, 1f, _e2068_);
                                if (_e2054_ != _e2054_) {
                                    phi_9103_ = true;
                                } else {
                                    phi_9103_ = (1f <= _e2054_);
                                }
                                let _e2073_ = phi_9103_;
                                let _e2074_ = select(_e2054_, 1f, _e2073_);
                                if (_e2059_ != _e2059_) {
                                    phi_9118_ = true;
                                } else {
                                    phi_9118_ = (1f <= _e2059_);
                                }
                                let _e2078_ = phi_9118_;
                                let _e2079_ = select(_e2059_, 1f, _e2078_);
                                if (_e2064_ != _e2064_) {
                                    phi_9133_ = true;
                                } else {
                                    phi_9133_ = (1f <= _e2064_);
                                }
                                let _e2083_ = phi_9133_;
                                let _e2084_ = select(_e2064_, 1f, _e2083_);
                                let _e2087_ = (_e538_ * _e1351_.member[0]);
                                let _e2090_ = (_e538_ * _e1351_.member[1]);
                                let _e2093_ = (_e538_ * _e1351_.member[2]);
                                let _e2096_ = (_e538_ * _e1351_.member[3]);
                                let _e2098_ = (_e1389_ - (_e1494_ * _e2087_));
                                let _e2100_ = (_e1389_ - (_e1494_ * _e2090_));
                                let _e2102_ = (_e1389_ - (_e1494_ * _e2093_));
                                let _e2104_ = (_e1389_ - (_e1494_ * _e2096_));
                                if (_e2098_ != _e2098_) {
                                    phi_9148_ = true;
                                } else {
                                    phi_9148_ = (-1f >= _e2098_);
                                }
                                let _e2108_ = phi_9148_;
                                let _e2109_ = select(_e2098_, -1f, _e2108_);
                                if (_e2100_ != _e2100_) {
                                    phi_9163_ = true;
                                } else {
                                    phi_9163_ = (-1f >= _e2100_);
                                }
                                let _e2113_ = phi_9163_;
                                let _e2114_ = select(_e2100_, -1f, _e2113_);
                                if (_e2102_ != _e2102_) {
                                    phi_9178_ = true;
                                } else {
                                    phi_9178_ = (-1f >= _e2102_);
                                }
                                let _e2118_ = phi_9178_;
                                let _e2119_ = select(_e2102_, -1f, _e2118_);
                                if (_e2104_ != _e2104_) {
                                    phi_9193_ = true;
                                } else {
                                    phi_9193_ = (-1f >= _e2104_);
                                }
                                let _e2123_ = phi_9193_;
                                let _e2124_ = select(_e2104_, -1f, _e2123_);
                                if (_e2109_ != _e2109_) {
                                    phi_9208_ = true;
                                } else {
                                    phi_9208_ = (1f <= _e2109_);
                                }
                                let _e2128_ = phi_9208_;
                                let _e2129_ = select(_e2109_, 1f, _e2128_);
                                if (_e2114_ != _e2114_) {
                                    phi_9223_ = true;
                                } else {
                                    phi_9223_ = (1f <= _e2114_);
                                }
                                let _e2133_ = phi_9223_;
                                let _e2134_ = select(_e2114_, 1f, _e2133_);
                                if (_e2119_ != _e2119_) {
                                    phi_9238_ = true;
                                } else {
                                    phi_9238_ = (1f <= _e2119_);
                                }
                                let _e2138_ = phi_9238_;
                                let _e2139_ = select(_e2119_, 1f, _e2138_);
                                if (_e2124_ != _e2124_) {
                                    phi_9253_ = true;
                                } else {
                                    phi_9253_ = (1f <= _e2124_);
                                }
                                let _e2143_ = phi_9253_;
                                let _e2144_ = select(_e2124_, 1f, _e2143_);
                                let _e2163_ = bitcast<f32>((bitcast<u32>(_e2098_) & 2147483647u));
                                let _e2166_ = bitcast<f32>((bitcast<u32>(_e2100_) & 2147483647u));
                                let _e2169_ = bitcast<f32>((bitcast<u32>(_e2102_) & 2147483647u));
                                let _e2172_ = bitcast<f32>((bitcast<u32>(_e2104_) & 2147483647u));
                                if (_e2163_ != _e2163_) {
                                    phi_9268_ = true;
                                } else {
                                    phi_9268_ = (1000000000000000f <= _e2163_);
                                }
                                let _e2176_ = phi_9268_;
                                let _e2177_ = select(_e2163_, 1000000000000000f, _e2176_);
                                if (_e2166_ != _e2166_) {
                                    phi_9283_ = true;
                                } else {
                                    phi_9283_ = (1000000000000000f <= _e2166_);
                                }
                                let _e2181_ = phi_9283_;
                                let _e2182_ = select(_e2166_, 1000000000000000f, _e2181_);
                                if (_e2169_ != _e2169_) {
                                    phi_9298_ = true;
                                } else {
                                    phi_9298_ = (1000000000000000f <= _e2169_);
                                }
                                let _e2186_ = phi_9298_;
                                let _e2187_ = select(_e2169_, 1000000000000000f, _e2186_);
                                if (_e2172_ != _e2172_) {
                                    phi_9313_ = true;
                                } else {
                                    phi_9313_ = (1000000000000000f <= _e2172_);
                                }
                                let _e2191_ = phi_9313_;
                                let _e2192_ = select(_e2172_, 1000000000000000f, _e2191_);
                                let _e2193_ = (_e2177_ * _e2177_);
                                let _e2194_ = (_e2182_ * _e2182_);
                                let _e2195_ = (_e2187_ * _e2187_);
                                let _e2196_ = (_e2192_ * _e2192_);
                                let _e2197_ = (_e2193_ + 0.0000010000001f);
                                let _e2198_ = (_e2194_ + 0.0000010000001f);
                                let _e2199_ = (_e2195_ + 0.0000010000001f);
                                let _e2200_ = (_e2196_ + 0.0000010000001f);
                                if (_e2197_ != _e2197_) {
                                    phi_9328_ = true;
                                } else {
                                    phi_9328_ = (0.000000000001f >= _e2197_);
                                }
                                let _e2204_ = phi_9328_;
                                let _e2205_ = select(_e2197_, 0.000000000001f, _e2204_);
                                if (_e2198_ != _e2198_) {
                                    phi_9343_ = true;
                                } else {
                                    phi_9343_ = (0.000000000001f >= _e2198_);
                                }
                                let _e2209_ = phi_9343_;
                                let _e2210_ = select(_e2198_, 0.000000000001f, _e2209_);
                                if (_e2199_ != _e2199_) {
                                    phi_9358_ = true;
                                } else {
                                    phi_9358_ = (0.000000000001f >= _e2199_);
                                }
                                let _e2214_ = phi_9358_;
                                let _e2215_ = select(_e2199_, 0.000000000001f, _e2214_);
                                if (_e2200_ != _e2200_) {
                                    phi_9373_ = true;
                                } else {
                                    phi_9373_ = (0.000000000001f >= _e2200_);
                                }
                                let _e2219_ = phi_9373_;
                                let _e2220_ = select(_e2200_, 0.000000000001f, _e2219_);
                                let _e2221_ = (_e2205_ * 0.5f);
                                let _e2222_ = (_e2210_ * 0.5f);
                                let _e2223_ = (_e2215_ * 0.5f);
                                let _e2224_ = (_e2220_ * 0.5f);
                                let _e2229_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2205_) >> bitcast<u32>(1i))));
                                let _e2234_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2210_) >> bitcast<u32>(1i))));
                                let _e2239_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2215_) >> bitcast<u32>(1i))));
                                let _e2244_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2220_) >> bitcast<u32>(1i))));
                                let _e2257_ = (_e2229_ * (1.5f - (_e2221_ * (_e2229_ * _e2229_))));
                                let _e2258_ = (_e2234_ * (1.5f - (_e2222_ * (_e2234_ * _e2234_))));
                                let _e2259_ = (_e2239_ * (1.5f - (_e2223_ * (_e2239_ * _e2239_))));
                                let _e2260_ = (_e2244_ * (1.5f - (_e2224_ * (_e2244_ * _e2244_))));
                                let _e2273_ = (_e2257_ * (1.5f - (_e2221_ * (_e2257_ * _e2257_))));
                                let _e2274_ = (_e2258_ * (1.5f - (_e2222_ * (_e2258_ * _e2258_))));
                                let _e2275_ = (_e2259_ * (1.5f - (_e2223_ * (_e2259_ * _e2259_))));
                                let _e2276_ = (_e2260_ * (1.5f - (_e2224_ * (_e2260_ * _e2260_))));
                                let _e2293_ = (_e2197_ * (_e2273_ * (1.5f - (_e2221_ * (_e2273_ * _e2273_)))));
                                let _e2294_ = (_e2198_ * (_e2274_ * (1.5f - (_e2222_ * (_e2274_ * _e2274_)))));
                                let _e2295_ = (_e2199_ * (_e2275_ * (1.5f - (_e2223_ * (_e2275_ * _e2275_)))));
                                let _e2296_ = (_e2200_ * (_e2276_ * (1.5f - (_e2224_ * (_e2276_ * _e2276_)))));
                                let _e2297_ = (_e2293_ + 0.001f);
                                let _e2298_ = (_e2294_ + 0.001f);
                                let _e2299_ = (_e2295_ + 0.001f);
                                let _e2300_ = (_e2296_ + 0.001f);
                                if (_e2297_ != _e2297_) {
                                    phi_9388_ = true;
                                } else {
                                    phi_9388_ = (0.000001f >= _e2297_);
                                }
                                let _e2304_ = phi_9388_;
                                let _e2305_ = select(_e2297_, 0.000001f, _e2304_);
                                if (_e2298_ != _e2298_) {
                                    phi_9403_ = true;
                                } else {
                                    phi_9403_ = (0.000001f >= _e2298_);
                                }
                                let _e2309_ = phi_9403_;
                                let _e2310_ = select(_e2298_, 0.000001f, _e2309_);
                                if (_e2299_ != _e2299_) {
                                    phi_9418_ = true;
                                } else {
                                    phi_9418_ = (0.000001f >= _e2299_);
                                }
                                let _e2314_ = phi_9418_;
                                let _e2315_ = select(_e2299_, 0.000001f, _e2314_);
                                if (_e2300_ != _e2300_) {
                                    phi_9433_ = true;
                                } else {
                                    phi_9433_ = (0.000001f >= _e2300_);
                                }
                                let _e2319_ = phi_9433_;
                                let _e2320_ = select(_e2300_, 0.000001f, _e2319_);
                                let _e2323_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2305_)));
                                let _e2326_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2310_)));
                                let _e2329_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2315_)));
                                let _e2332_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2320_)));
                                let _e2342_ = (_e2323_ + (_e2323_ * (1f - (_e2305_ * _e2323_))));
                                let _e2344_ = (_e2326_ + (_e2326_ * (1f - (_e2310_ * _e2326_))));
                                let _e2346_ = (_e2329_ + (_e2329_ * (1f - (_e2315_ * _e2329_))));
                                let _e2348_ = (_e2332_ + (_e2332_ * (1f - (_e2320_ * _e2332_))));
                                let _e2358_ = (_e2342_ + (_e2342_ * (1f - (_e2305_ * _e2342_))));
                                let _e2360_ = (_e2344_ + (_e2344_ * (1f - (_e2310_ * _e2344_))));
                                let _e2362_ = (_e2346_ + (_e2346_ * (1f - (_e2315_ * _e2346_))));
                                let _e2364_ = (_e2348_ + (_e2348_ * (1f - (_e2320_ * _e2348_))));
                                let _e2374_ = (_e2358_ + (_e2358_ * (1f - (_e2305_ * _e2358_))));
                                let _e2376_ = (_e2360_ + (_e2360_ * (1f - (_e2310_ * _e2360_))));
                                let _e2378_ = (_e2362_ + (_e2362_ * (1f - (_e2315_ * _e2362_))));
                                let _e2380_ = (_e2364_ + (_e2364_ * (1f - (_e2320_ * _e2364_))));
                                let _e2381_ = (_e2293_ + _e2177_);
                                let _e2382_ = (_e2294_ + _e2182_);
                                let _e2383_ = (_e2295_ + _e2187_);
                                let _e2384_ = (_e2296_ + _e2192_);
                                if (_e2381_ != _e2381_) {
                                    phi_9448_ = true;
                                } else {
                                    phi_9448_ = (0.000001f >= _e2381_);
                                }
                                let _e2388_ = phi_9448_;
                                let _e2389_ = select(_e2381_, 0.000001f, _e2388_);
                                if (_e2382_ != _e2382_) {
                                    phi_9463_ = true;
                                } else {
                                    phi_9463_ = (0.000001f >= _e2382_);
                                }
                                let _e2393_ = phi_9463_;
                                let _e2394_ = select(_e2382_, 0.000001f, _e2393_);
                                if (_e2383_ != _e2383_) {
                                    phi_9478_ = true;
                                } else {
                                    phi_9478_ = (0.000001f >= _e2383_);
                                }
                                let _e2398_ = phi_9478_;
                                let _e2399_ = select(_e2383_, 0.000001f, _e2398_);
                                if (_e2384_ != _e2384_) {
                                    phi_9493_ = true;
                                } else {
                                    phi_9493_ = (0.000001f >= _e2384_);
                                }
                                let _e2403_ = phi_9493_;
                                let _e2404_ = select(_e2384_, 0.000001f, _e2403_);
                                let _e2407_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2389_)));
                                let _e2410_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2394_)));
                                let _e2413_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2399_)));
                                let _e2416_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2404_)));
                                let _e2426_ = (_e2407_ + (_e2407_ * (1f - (_e2389_ * _e2407_))));
                                let _e2428_ = (_e2410_ + (_e2410_ * (1f - (_e2394_ * _e2410_))));
                                let _e2430_ = (_e2413_ + (_e2413_ * (1f - (_e2399_ * _e2413_))));
                                let _e2432_ = (_e2416_ + (_e2416_ * (1f - (_e2404_ * _e2416_))));
                                let _e2442_ = (_e2426_ + (_e2426_ * (1f - (_e2389_ * _e2426_))));
                                let _e2444_ = (_e2428_ + (_e2428_ * (1f - (_e2394_ * _e2428_))));
                                let _e2446_ = (_e2430_ + (_e2430_ * (1f - (_e2399_ * _e2430_))));
                                let _e2448_ = (_e2432_ + (_e2432_ * (1f - (_e2404_ * _e2432_))));
                                let _e2485_ = (_e2163_ <= 1000000000000000f);
                                let _e2487_ = (_e2166_ <= 1000000000000000f);
                                let _e2489_ = (_e2169_ <= 1000000000000000f);
                                let _e2491_ = (_e2172_ <= 1000000000000000f);
                                let _e2522_ = select(select(select(_e2163_, (0.5f * (_e2177_ + (_e2193_ * _e2374_))), _e2485_), select(-0.0005f, -(((0.0005f * (_e2177_ * (_e2442_ + (_e2442_ * (1f - (_e2389_ * _e2442_)))))) * (1f + (_e2177_ * _e2374_)))), _e2485_), (_e2098_ <= 0f)), 0f, (_e2163_ <= 0f));
                                let _e2524_ = select(select(select(_e2166_, (0.5f * (_e2182_ + (_e2194_ * _e2376_))), _e2487_), select(-0.0005f, -(((0.0005f * (_e2182_ * (_e2444_ + (_e2444_ * (1f - (_e2394_ * _e2444_)))))) * (1f + (_e2182_ * _e2376_)))), _e2487_), (_e2100_ <= 0f)), 0f, (_e2166_ <= 0f));
                                let _e2526_ = select(select(select(_e2169_, (0.5f * (_e2187_ + (_e2195_ * _e2378_))), _e2489_), select(-0.0005f, -(((0.0005f * (_e2187_ * (_e2446_ + (_e2446_ * (1f - (_e2399_ * _e2446_)))))) * (1f + (_e2187_ * _e2378_)))), _e2489_), (_e2102_ <= 0f)), 0f, (_e2169_ <= 0f));
                                let _e2528_ = select(select(select(_e2172_, (0.5f * (_e2192_ + (_e2196_ * _e2380_))), _e2491_), select(-0.0005f, -(((0.0005f * (_e2192_ * (_e2448_ + (_e2448_ * (1f - (_e2404_ * _e2448_)))))) * (1f + (_e2192_ * _e2380_)))), _e2491_), (_e2104_ <= 0f)), 0f, (_e2172_ <= 0f));
                                let _e2565_ = bitcast<u32>((((1f - (_e562_ * _e2069_)) * _e2087_) + (_e2069_ * select(((_e1493_ * _e2522_) + (_e562_ * (_e2129_ * (1f - (0.33333334f * (_e2129_ * _e2129_)))))), _e2522_, _e1939_))));
                                let _e2566_ = (_e2565_ & 2139095040u);
                                if (_e2566_ == 2139095040u) {
                                    phi_5771_ = true;
                                } else {
                                    phi_5771_ = (_e2566_ == 0u);
                                }
                                let _e2570_ = phi_5771_;
                                let _e2572_ = bitcast<f32>(select(_e2565_, 0u, _e2570_));
                                let _e2573_ = bitcast<u32>((((1f - (_e562_ * _e2074_)) * _e2090_) + (_e2074_ * select(((_e1493_ * _e2524_) + (_e562_ * (_e2134_ * (1f - (0.33333334f * (_e2134_ * _e2134_)))))), _e2524_, _e1939_))));
                                let _e2574_ = (_e2573_ & 2139095040u);
                                if (_e2574_ == 2139095040u) {
                                    phi_5787_ = true;
                                } else {
                                    phi_5787_ = (_e2574_ == 0u);
                                }
                                let _e2578_ = phi_5787_;
                                let _e2581_ = bitcast<u32>((((1f - (_e562_ * _e2079_)) * _e2093_) + (_e2079_ * select(((_e1493_ * _e2526_) + (_e562_ * (_e2139_ * (1f - (0.33333334f * (_e2139_ * _e2139_)))))), _e2526_, _e1939_))));
                                let _e2582_ = (_e2581_ & 2139095040u);
                                if (_e2582_ == 2139095040u) {
                                    phi_5803_ = true;
                                } else {
                                    phi_5803_ = (_e2582_ == 0u);
                                }
                                let _e2586_ = phi_5803_;
                                let _e2589_ = bitcast<u32>((((1f - (_e562_ * _e2084_)) * _e2096_) + (_e2084_ * select(((_e1493_ * _e2528_) + (_e562_ * (_e2144_ * (1f - (0.33333334f * (_e2144_ * _e2144_)))))), _e2528_, _e1939_))));
                                let _e2590_ = (_e2589_ & 2139095040u);
                                if (_e2590_ == 2139095040u) {
                                    phi_5819_ = true;
                                } else {
                                    phi_5819_ = (_e2590_ == 0u);
                                }
                                let _e2594_ = phi_5819_;
                                let _e2600_ = (0.5f * (_e1971_ + _e2572_));
                                let _e2607_ = (1024u + _e1368_.member_1_);
                                if (_e2607_ < 1536u) {
                                } else {
                                    phi_12159_ = true;
                                    break;
                                }
                                scan_scratch[_e2607_] = (_e1971_ + (_e568_ * (_e2600_ - _e1971_)));
                                let _e2610_ = (1280u + _e1368_.member_1_);
                                if (_e2610_ < 1536u) {
                                } else {
                                    phi_12159_ = true;
                                    break;
                                }
                                scan_scratch[_e2610_] = (_e2572_ + (_e568_ * (_e2600_ - _e2572_)));
                                phi_4112_ = type_22_(array<f32, 4>(_e2572_, bitcast<f32>(select(_e2573_, 0u, _e2578_)), bitcast<f32>(select(_e2581_, 0u, _e2586_)), bitcast<f32>(select(_e2589_, 0u, _e2594_))));
                                phi_4114_ = type_22_(array<f32, 4>(_e1971_, bitcast<f32>(select(_e1972_, 0u, _e1977_)), bitcast<f32>(select(_e1980_, 0u, _e1985_)), bitcast<f32>(select(_e1988_, 0u, _e1993_))));
                            } else {
                                phi_4112_ = type_22_();
                                phi_4114_ = type_22_();
                            }
                            let _e2614_ = phi_4112_;
                            let _e2616_ = phi_4114_;
                            phi_4108_ = _e1366_;
                            phi_4111_ = _e2614_;
                            phi_4113_ = _e2616_;
                            phi_12159_ = _e1304_;
                            if !(_e1372_) {
                                break;
                            }
                        }
                        let _e2619_ = phi_12159_;
                        phi_12444_ = _e2619_;
                        if _e2619_ {
                            break;
                        }
                        phi_12841_ = _e2619_;
                    } else {
                        phi_12841_ = _e1304_;
                    }
                    let _e2621_ = phi_12841_;
                    workgroupBarrier();
                    if _e158_ {
                        if _e883_ {
                        } else {
                            phi_12444_ = true;
                            break;
                        }
                        let _e2622_ = scan_scratch[_e882_];
                        phi_5907_ = _e2622_;
                    } else {
                        phi_5907_ = 0f;
                    }
                    let _e2624_ = phi_5907_;
                    if _e158_ {
                        if _e887_ {
                        } else {
                            phi_12444_ = true;
                            break;
                        }
                        let _e2625_ = scan_scratch[_e886_];
                        phi_5919_ = _e2625_;
                    } else {
                        phi_5919_ = 0f;
                    }
                    let _e2627_ = phi_5919_;
                    let _e2629_ = (1f - (_e773_ * _e2624_));
                    if (_e2629_ != _e2629_) {
                        phi_9628_ = true;
                    } else {
                        phi_9628_ = (0.05f >= _e2629_);
                    }
                    let _e2633_ = phi_9628_;
                    let _e2636_ = (1f - (_e773_ * _e2627_));
                    if (_e2636_ != _e2636_) {
                        phi_9643_ = true;
                    } else {
                        phi_9643_ = (0.05f >= _e2636_);
                    }
                    let _e2640_ = phi_9643_;
                    if _e158_ {
                        let _e2646_ = bitcast<u32>((((_e1106_.member * _e812_) * select(_e2629_, 0.05f, _e2633_)) + (_e851_ * _e2624_)));
                        let _e2647_ = (_e2646_ & 2139095040u);
                        if (_e2647_ == 2139095040u) {
                            phi_5943_ = true;
                        } else {
                            phi_5943_ = (_e2647_ == 0u);
                        }
                        let _e2651_ = phi_5943_;
                        phi_5949_ = bitcast<f32>(select(_e2646_, 0u, _e2651_));
                    } else {
                        phi_5949_ = 0f;
                    }
                    let _e2655_ = phi_5949_;
                    if _e158_ {
                        let _e2660_ = bitcast<u32>((((_e1343_.member * _e812_) * select(_e2636_, 0.05f, _e2640_)) + (_e851_ * _e2627_)));
                        let _e2661_ = (_e2660_ & 2139095040u);
                        if (_e2661_ == 2139095040u) {
                            phi_5967_ = true;
                        } else {
                            phi_5967_ = (_e2661_ == 0u);
                        }
                        let _e2665_ = phi_5967_;
                        phi_5973_ = bitcast<f32>(select(_e2660_, 0u, _e2665_));
                    } else {
                        phi_5973_ = 0f;
                    }
                    let _e2669_ = phi_5973_;
                    workgroupBarrier();
                    if _e867_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e865_] = _e2655_;
                    if _e871_ {
                    } else {
                        phi_12444_ = true;
                        break;
                    }
                    scan_scratch[_e870_] = _e2669_;
                    workgroupBarrier();
                    let _e2670_ = (_e113_ > 0u);
                    if _e2670_ {
                        if _e158_ {
                            let _e2671_ = (_e113_ - 1u);
                            if (_e2671_ < 1536u) {
                            } else {
                                phi_12444_ = true;
                                break;
                            }
                            let _e2674_ = scan_scratch[_e2671_];
                            phi_6002_ = _e2674_;
                        } else {
                            phi_6002_ = f32();
                        }
                        let _e2676_ = phi_6002_;
                        phi_6004_ = _e2676_;
                        phi_6005_ = select(true, false, _e158_);
                    } else {
                        phi_6004_ = f32();
                        phi_6005_ = true;
                    }
                    let _e2679_ = phi_6004_;
                    let _e2681_ = phi_6005_;
                    let _e2682_ = select(_e2679_, 0f, _e2681_);
                    if _e2670_ {
                        if _e158_ {
                            let _e2683_ = (_e113_ + 255u);
                            if (_e2683_ < 1536u) {
                            } else {
                                phi_12444_ = true;
                                break;
                            }
                            let _e2686_ = scan_scratch[_e2683_];
                            phi_6026_ = _e2686_;
                        } else {
                            phi_6026_ = f32();
                        }
                        let _e2688_ = phi_6026_;
                        phi_6028_ = _e2688_;
                        phi_6029_ = select(true, false, _e158_);
                    } else {
                        phi_6028_ = f32();
                        phi_6029_ = true;
                    }
                    let _e2691_ = phi_6028_;
                    let _e2693_ = phi_6029_;
                    let _e2694_ = select(_e2691_, 0f, _e2693_);
                    let _e2699_ = select(0f, _e2682_, (bitcast<f32>((bitcast<u32>(_e2682_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2700_ = (_e2699_ != _e2699_);
                    if _e2700_ {
                        phi_9658_ = true;
                    } else {
                        phi_9658_ = (-1000000000000000000f >= _e2699_);
                    }
                    let _e2703_ = phi_9658_;
                    let _e2704_ = select(_e2699_, -1000000000000000000f, _e2703_);
                    if _e2700_ {
                        phi_9673_ = true;
                    } else {
                        phi_9673_ = (-1000000000000000000f >= _e2699_);
                    }
                    let _e2707_ = phi_9673_;
                    let _e2708_ = select(_e2699_, -1000000000000000000f, _e2707_);
                    if _e2700_ {
                        phi_9688_ = true;
                    } else {
                        phi_9688_ = (-1000000000000000000f >= _e2699_);
                    }
                    let _e2711_ = phi_9688_;
                    let _e2712_ = select(_e2699_, -1000000000000000000f, _e2711_);
                    if _e2700_ {
                        phi_9703_ = true;
                    } else {
                        phi_9703_ = (-1000000000000000000f >= _e2699_);
                    }
                    let _e2715_ = phi_9703_;
                    let _e2716_ = select(_e2699_, -1000000000000000000f, _e2715_);
                    if (_e2704_ != _e2704_) {
                        phi_9718_ = true;
                    } else {
                        phi_9718_ = (1000000000000000000f <= _e2704_);
                    }
                    let _e2720_ = phi_9718_;
                    let _e2721_ = select(_e2704_, 1000000000000000000f, _e2720_);
                    if (_e2708_ != _e2708_) {
                        phi_9733_ = true;
                    } else {
                        phi_9733_ = (1000000000000000000f <= _e2708_);
                    }
                    let _e2725_ = phi_9733_;
                    let _e2726_ = select(_e2708_, 1000000000000000000f, _e2725_);
                    if (_e2712_ != _e2712_) {
                        phi_9748_ = true;
                    } else {
                        phi_9748_ = (1000000000000000000f <= _e2712_);
                    }
                    let _e2730_ = phi_9748_;
                    let _e2731_ = select(_e2712_, 1000000000000000000f, _e2730_);
                    if (_e2716_ != _e2716_) {
                        phi_9763_ = true;
                    } else {
                        phi_9763_ = (1000000000000000000f <= _e2716_);
                    }
                    let _e2735_ = phi_9763_;
                    let _e2736_ = select(_e2716_, 1000000000000000000f, _e2735_);
                    let _e2741_ = select(0f, _e2655_, (bitcast<f32>((bitcast<u32>(_e2655_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2742_ = (_e2741_ != _e2741_);
                    if _e2742_ {
                        phi_9778_ = true;
                    } else {
                        phi_9778_ = (-1000000000000000000f >= _e2741_);
                    }
                    let _e2745_ = phi_9778_;
                    let _e2746_ = select(_e2741_, -1000000000000000000f, _e2745_);
                    if _e2742_ {
                        phi_9793_ = true;
                    } else {
                        phi_9793_ = (-1000000000000000000f >= _e2741_);
                    }
                    let _e2749_ = phi_9793_;
                    let _e2750_ = select(_e2741_, -1000000000000000000f, _e2749_);
                    if _e2742_ {
                        phi_9808_ = true;
                    } else {
                        phi_9808_ = (-1000000000000000000f >= _e2741_);
                    }
                    let _e2753_ = phi_9808_;
                    let _e2754_ = select(_e2741_, -1000000000000000000f, _e2753_);
                    if _e2742_ {
                        phi_9823_ = true;
                    } else {
                        phi_9823_ = (-1000000000000000000f >= _e2741_);
                    }
                    let _e2757_ = phi_9823_;
                    let _e2758_ = select(_e2741_, -1000000000000000000f, _e2757_);
                    if (_e2746_ != _e2746_) {
                        phi_9838_ = true;
                    } else {
                        phi_9838_ = (1000000000000000000f <= _e2746_);
                    }
                    let _e2762_ = phi_9838_;
                    let _e2763_ = select(_e2746_, 1000000000000000000f, _e2762_);
                    if (_e2750_ != _e2750_) {
                        phi_9853_ = true;
                    } else {
                        phi_9853_ = (1000000000000000000f <= _e2750_);
                    }
                    let _e2767_ = phi_9853_;
                    let _e2768_ = select(_e2750_, 1000000000000000000f, _e2767_);
                    if (_e2754_ != _e2754_) {
                        phi_9868_ = true;
                    } else {
                        phi_9868_ = (1000000000000000000f <= _e2754_);
                    }
                    let _e2772_ = phi_9868_;
                    let _e2773_ = select(_e2754_, 1000000000000000000f, _e2772_);
                    if (_e2758_ != _e2758_) {
                        phi_9883_ = true;
                    } else {
                        phi_9883_ = (1000000000000000000f <= _e2758_);
                    }
                    let _e2777_ = phi_9883_;
                    let _e2778_ = select(_e2758_, 1000000000000000000f, _e2777_);
                    let _e2779_ = array<f32, 4>(_e2763_, _e2768_, _e2773_, _e2778_);
                    let _e2780_ = (_e2721_ != _e2721_);
                    if _e2780_ {
                        phi_9898_ = true;
                    } else {
                        phi_9898_ = (_e2763_ <= _e2721_);
                    }
                    let _e2783_ = phi_9898_;
                    let _e2784_ = select(_e2721_, _e2763_, _e2783_);
                    let _e2785_ = (_e2726_ != _e2726_);
                    if _e2785_ {
                        phi_9913_ = true;
                    } else {
                        phi_9913_ = (_e2768_ <= _e2726_);
                    }
                    let _e2788_ = phi_9913_;
                    let _e2789_ = select(_e2726_, _e2768_, _e2788_);
                    let _e2790_ = (_e2731_ != _e2731_);
                    if _e2790_ {
                        phi_9928_ = true;
                    } else {
                        phi_9928_ = (_e2773_ <= _e2731_);
                    }
                    let _e2793_ = phi_9928_;
                    let _e2794_ = select(_e2731_, _e2773_, _e2793_);
                    let _e2795_ = (_e2736_ != _e2736_);
                    if _e2795_ {
                        phi_9943_ = true;
                    } else {
                        phi_9943_ = (_e2778_ <= _e2736_);
                    }
                    let _e2798_ = phi_9943_;
                    let _e2799_ = select(_e2736_, _e2778_, _e2798_);
                    if _e2780_ {
                        phi_9958_ = true;
                    } else {
                        phi_9958_ = (_e2763_ >= _e2721_);
                    }
                    let _e2802_ = phi_9958_;
                    let _e2803_ = select(_e2721_, _e2763_, _e2802_);
                    if _e2785_ {
                        phi_9973_ = true;
                    } else {
                        phi_9973_ = (_e2768_ >= _e2726_);
                    }
                    let _e2806_ = phi_9973_;
                    let _e2807_ = select(_e2726_, _e2768_, _e2806_);
                    if _e2790_ {
                        phi_9988_ = true;
                    } else {
                        phi_9988_ = (_e2773_ >= _e2731_);
                    }
                    let _e2810_ = phi_9988_;
                    let _e2811_ = select(_e2731_, _e2773_, _e2810_);
                    if _e2795_ {
                        phi_10003_ = true;
                    } else {
                        phi_10003_ = (_e2778_ >= _e2736_);
                    }
                    let _e2814_ = phi_10003_;
                    let _e2815_ = select(_e2736_, _e2778_, _e2814_);
                    let _e2816_ = (_e2784_ != _e2784_);
                    if _e2816_ {
                        phi_10018_ = true;
                    } else {
                        phi_10018_ = (-1f >= _e2784_);
                    }
                    let _e2819_ = phi_10018_;
                    let _e2820_ = select(_e2784_, -1f, _e2819_);
                    let _e2821_ = (_e2789_ != _e2789_);
                    if _e2821_ {
                        phi_10033_ = true;
                    } else {
                        phi_10033_ = (-1f >= _e2789_);
                    }
                    let _e2824_ = phi_10033_;
                    let _e2825_ = select(_e2789_, -1f, _e2824_);
                    let _e2826_ = (_e2794_ != _e2794_);
                    if _e2826_ {
                        phi_10048_ = true;
                    } else {
                        phi_10048_ = (-1f >= _e2794_);
                    }
                    let _e2829_ = phi_10048_;
                    let _e2830_ = select(_e2794_, -1f, _e2829_);
                    let _e2831_ = (_e2799_ != _e2799_);
                    if _e2831_ {
                        phi_10063_ = true;
                    } else {
                        phi_10063_ = (-1f >= _e2799_);
                    }
                    let _e2834_ = phi_10063_;
                    let _e2835_ = select(_e2799_, -1f, _e2834_);
                    if (_e2820_ != _e2820_) {
                        phi_10078_ = true;
                    } else {
                        phi_10078_ = (1f <= _e2820_);
                    }
                    let _e2839_ = phi_10078_;
                    let _e2840_ = select(_e2820_, 1f, _e2839_);
                    if (_e2825_ != _e2825_) {
                        phi_10093_ = true;
                    } else {
                        phi_10093_ = (1f <= _e2825_);
                    }
                    let _e2844_ = phi_10093_;
                    let _e2845_ = select(_e2825_, 1f, _e2844_);
                    if (_e2830_ != _e2830_) {
                        phi_10108_ = true;
                    } else {
                        phi_10108_ = (1f <= _e2830_);
                    }
                    let _e2849_ = phi_10108_;
                    let _e2850_ = select(_e2830_, 1f, _e2849_);
                    if (_e2835_ != _e2835_) {
                        phi_10123_ = true;
                    } else {
                        phi_10123_ = (1f <= _e2835_);
                    }
                    let _e2854_ = phi_10123_;
                    let _e2855_ = select(_e2835_, 1f, _e2854_);
                    let _e2856_ = (_e2803_ != _e2803_);
                    if _e2856_ {
                        phi_10138_ = true;
                    } else {
                        phi_10138_ = (-1f >= _e2803_);
                    }
                    let _e2859_ = phi_10138_;
                    let _e2860_ = select(_e2803_, -1f, _e2859_);
                    let _e2861_ = (_e2807_ != _e2807_);
                    if _e2861_ {
                        phi_10153_ = true;
                    } else {
                        phi_10153_ = (-1f >= _e2807_);
                    }
                    let _e2864_ = phi_10153_;
                    let _e2865_ = select(_e2807_, -1f, _e2864_);
                    let _e2866_ = (_e2811_ != _e2811_);
                    if _e2866_ {
                        phi_10168_ = true;
                    } else {
                        phi_10168_ = (-1f >= _e2811_);
                    }
                    let _e2869_ = phi_10168_;
                    let _e2870_ = select(_e2811_, -1f, _e2869_);
                    let _e2871_ = (_e2815_ != _e2815_);
                    if _e2871_ {
                        phi_10183_ = true;
                    } else {
                        phi_10183_ = (-1f >= _e2815_);
                    }
                    let _e2874_ = phi_10183_;
                    let _e2875_ = select(_e2815_, -1f, _e2874_);
                    if (_e2860_ != _e2860_) {
                        phi_10198_ = true;
                    } else {
                        phi_10198_ = (1f <= _e2860_);
                    }
                    let _e2879_ = phi_10198_;
                    let _e2880_ = select(_e2860_, 1f, _e2879_);
                    if (_e2865_ != _e2865_) {
                        phi_10213_ = true;
                    } else {
                        phi_10213_ = (1f <= _e2865_);
                    }
                    let _e2884_ = phi_10213_;
                    let _e2885_ = select(_e2865_, 1f, _e2884_);
                    if (_e2870_ != _e2870_) {
                        phi_10228_ = true;
                    } else {
                        phi_10228_ = (1f <= _e2870_);
                    }
                    let _e2889_ = phi_10228_;
                    let _e2890_ = select(_e2870_, 1f, _e2889_);
                    if (_e2875_ != _e2875_) {
                        phi_10243_ = true;
                    } else {
                        phi_10243_ = (1f <= _e2875_);
                    }
                    let _e2894_ = phi_10243_;
                    let _e2895_ = select(_e2875_, 1f, _e2894_);
                    let _e2900_ = (0.5f * (_e2880_ + _e2840_));
                    let _e2901_ = (0.5f * (_e2885_ + _e2845_));
                    let _e2902_ = (0.5f * (_e2890_ + _e2850_));
                    let _e2903_ = (0.5f * (_e2895_ + _e2855_));
                    let _e2904_ = (_e2880_ - _e2840_);
                    let _e2905_ = (_e2885_ - _e2845_);
                    let _e2906_ = (_e2890_ - _e2850_);
                    let _e2907_ = (_e2895_ - _e2855_);
                    let _e2908_ = (0.5f * _e2904_);
                    let _e2909_ = (0.5f * _e2905_);
                    let _e2910_ = (0.5f * _e2906_);
                    let _e2911_ = (0.5f * _e2907_);
                    let _e2912_ = (_e2900_ * _e2900_);
                    let _e2913_ = (_e2901_ * _e2901_);
                    let _e2914_ = (_e2902_ * _e2902_);
                    let _e2915_ = (_e2903_ * _e2903_);
                    let _e2916_ = (_e2908_ * _e2908_);
                    let _e2917_ = (_e2909_ * _e2909_);
                    let _e2918_ = (_e2910_ * _e2910_);
                    let _e2919_ = (_e2911_ * _e2911_);
                    if (_e2904_ != _e2904_) {
                        phi_10258_ = true;
                    } else {
                        phi_10258_ = (0f >= _e2904_);
                    }
                    let _e2923_ = phi_10258_;
                    let _e2924_ = select(_e2904_, 0f, _e2923_);
                    if (_e2905_ != _e2905_) {
                        phi_10273_ = true;
                    } else {
                        phi_10273_ = (0f >= _e2905_);
                    }
                    let _e2928_ = phi_10273_;
                    let _e2929_ = select(_e2905_, 0f, _e2928_);
                    if (_e2906_ != _e2906_) {
                        phi_10288_ = true;
                    } else {
                        phi_10288_ = (0f >= _e2906_);
                    }
                    let _e2933_ = phi_10288_;
                    let _e2934_ = select(_e2906_, 0f, _e2933_);
                    if (_e2907_ != _e2907_) {
                        phi_10303_ = true;
                    } else {
                        phi_10303_ = (0f >= _e2907_);
                    }
                    let _e2938_ = phi_10303_;
                    let _e2939_ = select(_e2907_, 0f, _e2938_);
                    let _e2940_ = (_e2803_ - _e2784_);
                    let _e2941_ = (_e2807_ - _e2789_);
                    let _e2942_ = (_e2811_ - _e2794_);
                    let _e2943_ = (_e2815_ - _e2799_);
                    if _e2856_ {
                        phi_10318_ = true;
                    } else {
                        phi_10318_ = (-1f <= _e2803_);
                    }
                    let _e2946_ = phi_10318_;
                    if _e2861_ {
                        phi_10333_ = true;
                    } else {
                        phi_10333_ = (-1f <= _e2807_);
                    }
                    let _e2950_ = phi_10333_;
                    if _e2866_ {
                        phi_10348_ = true;
                    } else {
                        phi_10348_ = (-1f <= _e2811_);
                    }
                    let _e2954_ = phi_10348_;
                    if _e2871_ {
                        phi_10363_ = true;
                    } else {
                        phi_10363_ = (-1f <= _e2815_);
                    }
                    let _e2958_ = phi_10363_;
                    let _e2960_ = (select(_e2803_, -1f, _e2946_) - _e2784_);
                    let _e2961_ = (select(_e2807_, -1f, _e2950_) - _e2789_);
                    let _e2962_ = (select(_e2811_, -1f, _e2954_) - _e2794_);
                    let _e2963_ = (select(_e2815_, -1f, _e2958_) - _e2799_);
                    if (_e2960_ != _e2960_) {
                        phi_10378_ = true;
                    } else {
                        phi_10378_ = (0f >= _e2960_);
                    }
                    let _e2967_ = phi_10378_;
                    if (_e2961_ != _e2961_) {
                        phi_10393_ = true;
                    } else {
                        phi_10393_ = (0f >= _e2961_);
                    }
                    let _e2972_ = phi_10393_;
                    if (_e2962_ != _e2962_) {
                        phi_10408_ = true;
                    } else {
                        phi_10408_ = (0f >= _e2962_);
                    }
                    let _e2977_ = phi_10408_;
                    if (_e2963_ != _e2963_) {
                        phi_10423_ = true;
                    } else {
                        phi_10423_ = (0f >= _e2963_);
                    }
                    let _e2982_ = phi_10423_;
                    if _e2816_ {
                        phi_10438_ = true;
                    } else {
                        phi_10438_ = (1f >= _e2784_);
                    }
                    let _e2986_ = phi_10438_;
                    if _e2821_ {
                        phi_10453_ = true;
                    } else {
                        phi_10453_ = (1f >= _e2789_);
                    }
                    let _e2990_ = phi_10453_;
                    if _e2826_ {
                        phi_10468_ = true;
                    } else {
                        phi_10468_ = (1f >= _e2794_);
                    }
                    let _e2994_ = phi_10468_;
                    if _e2831_ {
                        phi_10483_ = true;
                    } else {
                        phi_10483_ = (1f >= _e2799_);
                    }
                    let _e2998_ = phi_10483_;
                    let _e3000_ = (_e2803_ - select(_e2784_, 1f, _e2986_));
                    let _e3001_ = (_e2807_ - select(_e2789_, 1f, _e2990_));
                    let _e3002_ = (_e2811_ - select(_e2794_, 1f, _e2994_));
                    let _e3003_ = (_e2815_ - select(_e2799_, 1f, _e2998_));
                    if (_e3000_ != _e3000_) {
                        phi_10498_ = true;
                    } else {
                        phi_10498_ = (0f >= _e3000_);
                    }
                    let _e3007_ = phi_10498_;
                    if (_e3001_ != _e3001_) {
                        phi_10513_ = true;
                    } else {
                        phi_10513_ = (0f >= _e3001_);
                    }
                    let _e3012_ = phi_10513_;
                    if (_e3002_ != _e3002_) {
                        phi_10528_ = true;
                    } else {
                        phi_10528_ = (0f >= _e3002_);
                    }
                    let _e3017_ = phi_10528_;
                    if (_e3003_ != _e3003_) {
                        phi_10543_ = true;
                    } else {
                        phi_10543_ = (0f >= _e3003_);
                    }
                    let _e3022_ = phi_10543_;
                    let _e3026_ = bitcast<f32>((bitcast<u32>(_e2721_) & 2147483647u));
                    let _e3029_ = bitcast<f32>((bitcast<u32>(_e2726_) & 2147483647u));
                    let _e3032_ = bitcast<f32>((bitcast<u32>(_e2731_) & 2147483647u));
                    let _e3035_ = bitcast<f32>((bitcast<u32>(_e2736_) & 2147483647u));
                    let _e3038_ = bitcast<f32>((bitcast<u32>(_e2763_) & 2147483647u));
                    let _e3041_ = bitcast<f32>((bitcast<u32>(_e2768_) & 2147483647u));
                    let _e3044_ = bitcast<f32>((bitcast<u32>(_e2773_) & 2147483647u));
                    let _e3047_ = bitcast<f32>((bitcast<u32>(_e2778_) & 2147483647u));
                    if (_e3026_ != _e3026_) {
                        phi_10558_ = true;
                    } else {
                        phi_10558_ = (_e3038_ >= _e3026_);
                    }
                    let _e3051_ = phi_10558_;
                    let _e3052_ = select(_e3026_, _e3038_, _e3051_);
                    if (_e3029_ != _e3029_) {
                        phi_10573_ = true;
                    } else {
                        phi_10573_ = (_e3041_ >= _e3029_);
                    }
                    let _e3056_ = phi_10573_;
                    let _e3057_ = select(_e3029_, _e3041_, _e3056_);
                    if (_e3032_ != _e3032_) {
                        phi_10588_ = true;
                    } else {
                        phi_10588_ = (_e3044_ >= _e3032_);
                    }
                    let _e3061_ = phi_10588_;
                    let _e3062_ = select(_e3032_, _e3044_, _e3061_);
                    if (_e3035_ != _e3035_) {
                        phi_10603_ = true;
                    } else {
                        phi_10603_ = (_e3047_ >= _e3035_);
                    }
                    let _e3066_ = phi_10603_;
                    let _e3067_ = select(_e3035_, _e3047_, _e3066_);
                    let _e3084_ = (_e3052_ <= 0f);
                    let _e3085_ = select((_e2900_ * (1.5f - (0.5f * (_e2912_ + _e2916_)))), 0f, _e3084_);
                    let _e3086_ = (_e3057_ <= 0f);
                    let _e3087_ = select((_e2901_ * (1.5f - (0.5f * (_e2913_ + _e2917_)))), 0f, _e3086_);
                    let _e3088_ = (_e3062_ <= 0f);
                    let _e3089_ = select((_e2902_ * (1.5f - (0.5f * (_e2914_ + _e2918_)))), 0f, _e3088_);
                    let _e3090_ = (_e3067_ <= 0f);
                    let _e3091_ = select((_e2903_ * (1.5f - (0.5f * (_e2915_ + _e2919_)))), 0f, _e3090_);
                    let _e3133_ = select(((_e2912_ + (0.33333334f * _e2916_)) - (((_e2912_ * _e2912_) + ((2f * _e2912_) * _e2916_)) + (0.2f * (_e2916_ * _e2916_)))), 0f, _e3084_);
                    let _e3134_ = select(((_e2913_ + (0.33333334f * _e2917_)) - (((_e2913_ * _e2913_) + ((2f * _e2913_) * _e2917_)) + (0.2f * (_e2917_ * _e2917_)))), 0f, _e3086_);
                    let _e3135_ = select(((_e2914_ + (0.33333334f * _e2918_)) - (((_e2914_ * _e2914_) + ((2f * _e2914_) * _e2918_)) + (0.2f * (_e2918_ * _e2918_)))), 0f, _e3088_);
                    let _e3136_ = select(((_e2915_ + (0.33333334f * _e2919_)) - (((_e2915_ * _e2915_) + ((2f * _e2915_) * _e2919_)) + (0.2f * (_e2919_ * _e2919_)))), 0f, _e3090_);
                    if (_e3052_ <= 1f) {
                        if (_e3057_ <= 1f) {
                            let _e3140_ = (_e3062_ <= 1f);
                            if _e3140_ {
                                phi_6456_ = (_e3067_ <= 1f);
                            } else {
                                phi_6456_ = bool();
                            }
                            let _e3143_ = phi_6456_;
                            phi_6458_ = _e3143_;
                            phi_6459_ = select(true, false, _e3140_);
                        } else {
                            phi_6458_ = bool();
                            phi_6459_ = true;
                        }
                        let _e3146_ = phi_6458_;
                        let _e3148_ = phi_6459_;
                        phi_6460_ = _e3146_;
                        phi_6461_ = _e3148_;
                    } else {
                        phi_6460_ = bool();
                        phi_6461_ = true;
                    }
                    let _e3150_ = phi_6460_;
                    let _e3152_ = phi_6461_;
                    if select(_e3150_, false, _e3152_) {
                        phi_6839_ = type_27_(type_22_(array<f32, 4>(_e3085_, _e3087_, _e3089_, _e3091_)), type_22_(array<f32, 4>(_e3133_, _e3134_, _e3135_, _e3136_)), type_22_(_e2779_));
                    } else {
                        if (_e2940_ != _e2940_) {
                            phi_10618_ = true;
                        } else {
                            phi_10618_ = (0.000001f >= _e2940_);
                        }
                        let _e3157_ = phi_10618_;
                        let _e3158_ = select(_e2940_, 0.000001f, _e3157_);
                        if (_e2941_ != _e2941_) {
                            phi_10633_ = true;
                        } else {
                            phi_10633_ = (0.000001f >= _e2941_);
                        }
                        let _e3162_ = phi_10633_;
                        let _e3163_ = select(_e2941_, 0.000001f, _e3162_);
                        if (_e2942_ != _e2942_) {
                            phi_10648_ = true;
                        } else {
                            phi_10648_ = (0.000001f >= _e2942_);
                        }
                        let _e3167_ = phi_10648_;
                        let _e3168_ = select(_e2942_, 0.000001f, _e3167_);
                        if (_e2943_ != _e2943_) {
                            phi_10663_ = true;
                        } else {
                            phi_10663_ = (0.000001f >= _e2943_);
                        }
                        let _e3172_ = phi_10663_;
                        let _e3173_ = select(_e2943_, 0.000001f, _e3172_);
                        let _e3176_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3158_)));
                        let _e3179_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3163_)));
                        let _e3182_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3168_)));
                        let _e3185_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3173_)));
                        let _e3195_ = (_e3176_ + (_e3176_ * (1f - (_e3158_ * _e3176_))));
                        let _e3197_ = (_e3179_ + (_e3179_ * (1f - (_e3163_ * _e3179_))));
                        let _e3199_ = (_e3182_ + (_e3182_ * (1f - (_e3168_ * _e3182_))));
                        let _e3201_ = (_e3185_ + (_e3185_ * (1f - (_e3173_ * _e3185_))));
                        let _e3211_ = (_e3195_ + (_e3195_ * (1f - (_e3158_ * _e3195_))));
                        let _e3213_ = (_e3197_ + (_e3197_ * (1f - (_e3163_ * _e3197_))));
                        let _e3215_ = (_e3199_ + (_e3199_ * (1f - (_e3168_ * _e3199_))));
                        let _e3217_ = (_e3201_ + (_e3201_ * (1f - (_e3173_ * _e3201_))));
                        let _e3270_ = (select((select(0f, 1f, (-1f <= _e2784_)) * select(0f, 1f, (_e2803_ <= 1f))), 1f, (_e2940_ <= 0.000001f)) <= 0f);
                        let _e3272_ = (select((select(0f, 1f, (-1f <= _e2789_)) * select(0f, 1f, (_e2807_ <= 1f))), 1f, (_e2941_ <= 0.000001f)) <= 0f);
                        let _e3274_ = (select((select(0f, 1f, (-1f <= _e2794_)) * select(0f, 1f, (_e2811_ <= 1f))), 1f, (_e2942_ <= 0.000001f)) <= 0f);
                        let _e3276_ = (select((select(0f, 1f, (-1f <= _e2799_)) * select(0f, 1f, (_e2815_ <= 1f))), 1f, (_e2943_ <= 0.000001f)) <= 0f);
                        let _e3282_ = (1f <= _e2784_);
                        let _e3284_ = (1f <= _e2789_);
                        let _e3286_ = (1f <= _e2794_);
                        let _e3288_ = (1f <= _e2799_);
                        let _e3290_ = (_e2803_ <= -1f);
                        let _e3292_ = (_e2807_ <= -1f);
                        let _e3294_ = (_e2811_ <= -1f);
                        let _e3296_ = (_e2815_ <= -1f);
                        phi_6839_ = type_27_(type_22_(array<f32, 4>(select(select(select(_e3085_, (((select(_e3000_, 0f, _e3007_) - select(_e2960_, 0f, _e2967_)) + (_e2924_ * _e3085_)) * _e3211_), _e3270_), 1f, _e3282_), -1f, _e3290_), select(select(select(_e3087_, (((select(_e3001_, 0f, _e3012_) - select(_e2961_, 0f, _e2972_)) + (_e2929_ * _e3087_)) * _e3213_), _e3272_), 1f, _e3284_), -1f, _e3292_), select(select(select(_e3089_, (((select(_e3002_, 0f, _e3017_) - select(_e2962_, 0f, _e2977_)) + (_e2934_ * _e3089_)) * _e3215_), _e3274_), 1f, _e3286_), -1f, _e3294_), select(select(select(_e3091_, (((select(_e3003_, 0f, _e3022_) - select(_e2963_, 0f, _e2982_)) + (_e2939_ * _e3091_)) * _e3217_), _e3276_), 1f, _e3288_), -1f, _e3296_))), type_22_(array<f32, 4>(select(select(select(_e3133_, ((_e2924_ * _e3133_) * _e3211_), _e3270_), 0f, _e3282_), 0f, _e3290_), select(select(select(_e3134_, ((_e2929_ * _e3134_) * _e3213_), _e3272_), 0f, _e3284_), 0f, _e3292_), select(select(select(_e3135_, ((_e2934_ * _e3135_) * _e3215_), _e3274_), 0f, _e3286_), 0f, _e3294_), select(select(select(_e3136_, ((_e2939_ * _e3136_) * _e3217_), _e3276_), 0f, _e3288_), 0f, _e3296_))), type_22_(_e2779_));
                    }
                    let _e3317_ = phi_6839_;
                    let _e3324_ = select(0f, _e2694_, (bitcast<f32>((bitcast<u32>(_e2694_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3325_ = (_e3324_ != _e3324_);
                    if _e3325_ {
                        phi_10678_ = true;
                    } else {
                        phi_10678_ = (-1000000000000000000f >= _e3324_);
                    }
                    let _e3328_ = phi_10678_;
                    let _e3329_ = select(_e3324_, -1000000000000000000f, _e3328_);
                    if _e3325_ {
                        phi_10693_ = true;
                    } else {
                        phi_10693_ = (-1000000000000000000f >= _e3324_);
                    }
                    let _e3332_ = phi_10693_;
                    let _e3333_ = select(_e3324_, -1000000000000000000f, _e3332_);
                    if _e3325_ {
                        phi_10708_ = true;
                    } else {
                        phi_10708_ = (-1000000000000000000f >= _e3324_);
                    }
                    let _e3336_ = phi_10708_;
                    let _e3337_ = select(_e3324_, -1000000000000000000f, _e3336_);
                    if _e3325_ {
                        phi_10723_ = true;
                    } else {
                        phi_10723_ = (-1000000000000000000f >= _e3324_);
                    }
                    let _e3340_ = phi_10723_;
                    let _e3341_ = select(_e3324_, -1000000000000000000f, _e3340_);
                    if (_e3329_ != _e3329_) {
                        phi_10738_ = true;
                    } else {
                        phi_10738_ = (1000000000000000000f <= _e3329_);
                    }
                    let _e3345_ = phi_10738_;
                    let _e3346_ = select(_e3329_, 1000000000000000000f, _e3345_);
                    if (_e3333_ != _e3333_) {
                        phi_10753_ = true;
                    } else {
                        phi_10753_ = (1000000000000000000f <= _e3333_);
                    }
                    let _e3350_ = phi_10753_;
                    let _e3351_ = select(_e3333_, 1000000000000000000f, _e3350_);
                    if (_e3337_ != _e3337_) {
                        phi_10768_ = true;
                    } else {
                        phi_10768_ = (1000000000000000000f <= _e3337_);
                    }
                    let _e3355_ = phi_10768_;
                    let _e3356_ = select(_e3337_, 1000000000000000000f, _e3355_);
                    if (_e3341_ != _e3341_) {
                        phi_10783_ = true;
                    } else {
                        phi_10783_ = (1000000000000000000f <= _e3341_);
                    }
                    let _e3360_ = phi_10783_;
                    let _e3361_ = select(_e3341_, 1000000000000000000f, _e3360_);
                    let _e3366_ = select(0f, _e2669_, (bitcast<f32>((bitcast<u32>(_e2669_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3367_ = (_e3366_ != _e3366_);
                    if _e3367_ {
                        phi_10798_ = true;
                    } else {
                        phi_10798_ = (-1000000000000000000f >= _e3366_);
                    }
                    let _e3370_ = phi_10798_;
                    let _e3371_ = select(_e3366_, -1000000000000000000f, _e3370_);
                    if _e3367_ {
                        phi_10813_ = true;
                    } else {
                        phi_10813_ = (-1000000000000000000f >= _e3366_);
                    }
                    let _e3374_ = phi_10813_;
                    let _e3375_ = select(_e3366_, -1000000000000000000f, _e3374_);
                    if _e3367_ {
                        phi_10828_ = true;
                    } else {
                        phi_10828_ = (-1000000000000000000f >= _e3366_);
                    }
                    let _e3378_ = phi_10828_;
                    let _e3379_ = select(_e3366_, -1000000000000000000f, _e3378_);
                    if _e3367_ {
                        phi_10843_ = true;
                    } else {
                        phi_10843_ = (-1000000000000000000f >= _e3366_);
                    }
                    let _e3382_ = phi_10843_;
                    let _e3383_ = select(_e3366_, -1000000000000000000f, _e3382_);
                    if (_e3371_ != _e3371_) {
                        phi_10858_ = true;
                    } else {
                        phi_10858_ = (1000000000000000000f <= _e3371_);
                    }
                    let _e3387_ = phi_10858_;
                    let _e3388_ = select(_e3371_, 1000000000000000000f, _e3387_);
                    if (_e3375_ != _e3375_) {
                        phi_10873_ = true;
                    } else {
                        phi_10873_ = (1000000000000000000f <= _e3375_);
                    }
                    let _e3392_ = phi_10873_;
                    let _e3393_ = select(_e3375_, 1000000000000000000f, _e3392_);
                    if (_e3379_ != _e3379_) {
                        phi_10888_ = true;
                    } else {
                        phi_10888_ = (1000000000000000000f <= _e3379_);
                    }
                    let _e3397_ = phi_10888_;
                    let _e3398_ = select(_e3379_, 1000000000000000000f, _e3397_);
                    if (_e3383_ != _e3383_) {
                        phi_10903_ = true;
                    } else {
                        phi_10903_ = (1000000000000000000f <= _e3383_);
                    }
                    let _e3402_ = phi_10903_;
                    let _e3403_ = select(_e3383_, 1000000000000000000f, _e3402_);
                    let _e3404_ = array<f32, 4>(_e3388_, _e3393_, _e3398_, _e3403_);
                    let _e3405_ = (_e3346_ != _e3346_);
                    if _e3405_ {
                        phi_10918_ = true;
                    } else {
                        phi_10918_ = (_e3388_ <= _e3346_);
                    }
                    let _e3408_ = phi_10918_;
                    let _e3409_ = select(_e3346_, _e3388_, _e3408_);
                    let _e3410_ = (_e3351_ != _e3351_);
                    if _e3410_ {
                        phi_10933_ = true;
                    } else {
                        phi_10933_ = (_e3393_ <= _e3351_);
                    }
                    let _e3413_ = phi_10933_;
                    let _e3414_ = select(_e3351_, _e3393_, _e3413_);
                    let _e3415_ = (_e3356_ != _e3356_);
                    if _e3415_ {
                        phi_10948_ = true;
                    } else {
                        phi_10948_ = (_e3398_ <= _e3356_);
                    }
                    let _e3418_ = phi_10948_;
                    let _e3419_ = select(_e3356_, _e3398_, _e3418_);
                    let _e3420_ = (_e3361_ != _e3361_);
                    if _e3420_ {
                        phi_10963_ = true;
                    } else {
                        phi_10963_ = (_e3403_ <= _e3361_);
                    }
                    let _e3423_ = phi_10963_;
                    let _e3424_ = select(_e3361_, _e3403_, _e3423_);
                    if _e3405_ {
                        phi_10978_ = true;
                    } else {
                        phi_10978_ = (_e3388_ >= _e3346_);
                    }
                    let _e3427_ = phi_10978_;
                    let _e3428_ = select(_e3346_, _e3388_, _e3427_);
                    if _e3410_ {
                        phi_10993_ = true;
                    } else {
                        phi_10993_ = (_e3393_ >= _e3351_);
                    }
                    let _e3431_ = phi_10993_;
                    let _e3432_ = select(_e3351_, _e3393_, _e3431_);
                    if _e3415_ {
                        phi_11008_ = true;
                    } else {
                        phi_11008_ = (_e3398_ >= _e3356_);
                    }
                    let _e3435_ = phi_11008_;
                    let _e3436_ = select(_e3356_, _e3398_, _e3435_);
                    if _e3420_ {
                        phi_11023_ = true;
                    } else {
                        phi_11023_ = (_e3403_ >= _e3361_);
                    }
                    let _e3439_ = phi_11023_;
                    let _e3440_ = select(_e3361_, _e3403_, _e3439_);
                    let _e3441_ = (_e3409_ != _e3409_);
                    if _e3441_ {
                        phi_11038_ = true;
                    } else {
                        phi_11038_ = (-1f >= _e3409_);
                    }
                    let _e3444_ = phi_11038_;
                    let _e3445_ = select(_e3409_, -1f, _e3444_);
                    let _e3446_ = (_e3414_ != _e3414_);
                    if _e3446_ {
                        phi_11053_ = true;
                    } else {
                        phi_11053_ = (-1f >= _e3414_);
                    }
                    let _e3449_ = phi_11053_;
                    let _e3450_ = select(_e3414_, -1f, _e3449_);
                    let _e3451_ = (_e3419_ != _e3419_);
                    if _e3451_ {
                        phi_11068_ = true;
                    } else {
                        phi_11068_ = (-1f >= _e3419_);
                    }
                    let _e3454_ = phi_11068_;
                    let _e3455_ = select(_e3419_, -1f, _e3454_);
                    let _e3456_ = (_e3424_ != _e3424_);
                    if _e3456_ {
                        phi_11083_ = true;
                    } else {
                        phi_11083_ = (-1f >= _e3424_);
                    }
                    let _e3459_ = phi_11083_;
                    let _e3460_ = select(_e3424_, -1f, _e3459_);
                    if (_e3445_ != _e3445_) {
                        phi_11098_ = true;
                    } else {
                        phi_11098_ = (1f <= _e3445_);
                    }
                    let _e3464_ = phi_11098_;
                    let _e3465_ = select(_e3445_, 1f, _e3464_);
                    if (_e3450_ != _e3450_) {
                        phi_11113_ = true;
                    } else {
                        phi_11113_ = (1f <= _e3450_);
                    }
                    let _e3469_ = phi_11113_;
                    let _e3470_ = select(_e3450_, 1f, _e3469_);
                    if (_e3455_ != _e3455_) {
                        phi_11128_ = true;
                    } else {
                        phi_11128_ = (1f <= _e3455_);
                    }
                    let _e3474_ = phi_11128_;
                    let _e3475_ = select(_e3455_, 1f, _e3474_);
                    if (_e3460_ != _e3460_) {
                        phi_11143_ = true;
                    } else {
                        phi_11143_ = (1f <= _e3460_);
                    }
                    let _e3479_ = phi_11143_;
                    let _e3480_ = select(_e3460_, 1f, _e3479_);
                    let _e3481_ = (_e3428_ != _e3428_);
                    if _e3481_ {
                        phi_11158_ = true;
                    } else {
                        phi_11158_ = (-1f >= _e3428_);
                    }
                    let _e3484_ = phi_11158_;
                    let _e3485_ = select(_e3428_, -1f, _e3484_);
                    let _e3486_ = (_e3432_ != _e3432_);
                    if _e3486_ {
                        phi_11173_ = true;
                    } else {
                        phi_11173_ = (-1f >= _e3432_);
                    }
                    let _e3489_ = phi_11173_;
                    let _e3490_ = select(_e3432_, -1f, _e3489_);
                    let _e3491_ = (_e3436_ != _e3436_);
                    if _e3491_ {
                        phi_11188_ = true;
                    } else {
                        phi_11188_ = (-1f >= _e3436_);
                    }
                    let _e3494_ = phi_11188_;
                    let _e3495_ = select(_e3436_, -1f, _e3494_);
                    let _e3496_ = (_e3440_ != _e3440_);
                    if _e3496_ {
                        phi_11203_ = true;
                    } else {
                        phi_11203_ = (-1f >= _e3440_);
                    }
                    let _e3499_ = phi_11203_;
                    let _e3500_ = select(_e3440_, -1f, _e3499_);
                    if (_e3485_ != _e3485_) {
                        phi_11218_ = true;
                    } else {
                        phi_11218_ = (1f <= _e3485_);
                    }
                    let _e3504_ = phi_11218_;
                    let _e3505_ = select(_e3485_, 1f, _e3504_);
                    if (_e3490_ != _e3490_) {
                        phi_11233_ = true;
                    } else {
                        phi_11233_ = (1f <= _e3490_);
                    }
                    let _e3509_ = phi_11233_;
                    let _e3510_ = select(_e3490_, 1f, _e3509_);
                    if (_e3495_ != _e3495_) {
                        phi_11248_ = true;
                    } else {
                        phi_11248_ = (1f <= _e3495_);
                    }
                    let _e3514_ = phi_11248_;
                    let _e3515_ = select(_e3495_, 1f, _e3514_);
                    if (_e3500_ != _e3500_) {
                        phi_11263_ = true;
                    } else {
                        phi_11263_ = (1f <= _e3500_);
                    }
                    let _e3519_ = phi_11263_;
                    let _e3520_ = select(_e3500_, 1f, _e3519_);
                    let _e3525_ = (0.5f * (_e3505_ + _e3465_));
                    let _e3526_ = (0.5f * (_e3510_ + _e3470_));
                    let _e3527_ = (0.5f * (_e3515_ + _e3475_));
                    let _e3528_ = (0.5f * (_e3520_ + _e3480_));
                    let _e3529_ = (_e3505_ - _e3465_);
                    let _e3530_ = (_e3510_ - _e3470_);
                    let _e3531_ = (_e3515_ - _e3475_);
                    let _e3532_ = (_e3520_ - _e3480_);
                    let _e3533_ = (0.5f * _e3529_);
                    let _e3534_ = (0.5f * _e3530_);
                    let _e3535_ = (0.5f * _e3531_);
                    let _e3536_ = (0.5f * _e3532_);
                    let _e3537_ = (_e3525_ * _e3525_);
                    let _e3538_ = (_e3526_ * _e3526_);
                    let _e3539_ = (_e3527_ * _e3527_);
                    let _e3540_ = (_e3528_ * _e3528_);
                    let _e3541_ = (_e3533_ * _e3533_);
                    let _e3542_ = (_e3534_ * _e3534_);
                    let _e3543_ = (_e3535_ * _e3535_);
                    let _e3544_ = (_e3536_ * _e3536_);
                    if (_e3529_ != _e3529_) {
                        phi_11278_ = true;
                    } else {
                        phi_11278_ = (0f >= _e3529_);
                    }
                    let _e3548_ = phi_11278_;
                    let _e3549_ = select(_e3529_, 0f, _e3548_);
                    if (_e3530_ != _e3530_) {
                        phi_11293_ = true;
                    } else {
                        phi_11293_ = (0f >= _e3530_);
                    }
                    let _e3553_ = phi_11293_;
                    let _e3554_ = select(_e3530_, 0f, _e3553_);
                    if (_e3531_ != _e3531_) {
                        phi_11308_ = true;
                    } else {
                        phi_11308_ = (0f >= _e3531_);
                    }
                    let _e3558_ = phi_11308_;
                    let _e3559_ = select(_e3531_, 0f, _e3558_);
                    if (_e3532_ != _e3532_) {
                        phi_11323_ = true;
                    } else {
                        phi_11323_ = (0f >= _e3532_);
                    }
                    let _e3563_ = phi_11323_;
                    let _e3564_ = select(_e3532_, 0f, _e3563_);
                    let _e3565_ = (_e3428_ - _e3409_);
                    let _e3566_ = (_e3432_ - _e3414_);
                    let _e3567_ = (_e3436_ - _e3419_);
                    let _e3568_ = (_e3440_ - _e3424_);
                    if _e3481_ {
                        phi_11338_ = true;
                    } else {
                        phi_11338_ = (-1f <= _e3428_);
                    }
                    let _e3571_ = phi_11338_;
                    if _e3486_ {
                        phi_11353_ = true;
                    } else {
                        phi_11353_ = (-1f <= _e3432_);
                    }
                    let _e3575_ = phi_11353_;
                    if _e3491_ {
                        phi_11368_ = true;
                    } else {
                        phi_11368_ = (-1f <= _e3436_);
                    }
                    let _e3579_ = phi_11368_;
                    if _e3496_ {
                        phi_11383_ = true;
                    } else {
                        phi_11383_ = (-1f <= _e3440_);
                    }
                    let _e3583_ = phi_11383_;
                    let _e3585_ = (select(_e3428_, -1f, _e3571_) - _e3409_);
                    let _e3586_ = (select(_e3432_, -1f, _e3575_) - _e3414_);
                    let _e3587_ = (select(_e3436_, -1f, _e3579_) - _e3419_);
                    let _e3588_ = (select(_e3440_, -1f, _e3583_) - _e3424_);
                    if (_e3585_ != _e3585_) {
                        phi_11398_ = true;
                    } else {
                        phi_11398_ = (0f >= _e3585_);
                    }
                    let _e3592_ = phi_11398_;
                    if (_e3586_ != _e3586_) {
                        phi_11413_ = true;
                    } else {
                        phi_11413_ = (0f >= _e3586_);
                    }
                    let _e3597_ = phi_11413_;
                    if (_e3587_ != _e3587_) {
                        phi_11428_ = true;
                    } else {
                        phi_11428_ = (0f >= _e3587_);
                    }
                    let _e3602_ = phi_11428_;
                    if (_e3588_ != _e3588_) {
                        phi_11443_ = true;
                    } else {
                        phi_11443_ = (0f >= _e3588_);
                    }
                    let _e3607_ = phi_11443_;
                    if _e3441_ {
                        phi_11458_ = true;
                    } else {
                        phi_11458_ = (1f >= _e3409_);
                    }
                    let _e3611_ = phi_11458_;
                    if _e3446_ {
                        phi_11473_ = true;
                    } else {
                        phi_11473_ = (1f >= _e3414_);
                    }
                    let _e3615_ = phi_11473_;
                    if _e3451_ {
                        phi_11488_ = true;
                    } else {
                        phi_11488_ = (1f >= _e3419_);
                    }
                    let _e3619_ = phi_11488_;
                    if _e3456_ {
                        phi_11503_ = true;
                    } else {
                        phi_11503_ = (1f >= _e3424_);
                    }
                    let _e3623_ = phi_11503_;
                    let _e3625_ = (_e3428_ - select(_e3409_, 1f, _e3611_));
                    let _e3626_ = (_e3432_ - select(_e3414_, 1f, _e3615_));
                    let _e3627_ = (_e3436_ - select(_e3419_, 1f, _e3619_));
                    let _e3628_ = (_e3440_ - select(_e3424_, 1f, _e3623_));
                    if (_e3625_ != _e3625_) {
                        phi_11518_ = true;
                    } else {
                        phi_11518_ = (0f >= _e3625_);
                    }
                    let _e3632_ = phi_11518_;
                    if (_e3626_ != _e3626_) {
                        phi_11533_ = true;
                    } else {
                        phi_11533_ = (0f >= _e3626_);
                    }
                    let _e3637_ = phi_11533_;
                    if (_e3627_ != _e3627_) {
                        phi_11548_ = true;
                    } else {
                        phi_11548_ = (0f >= _e3627_);
                    }
                    let _e3642_ = phi_11548_;
                    if (_e3628_ != _e3628_) {
                        phi_11563_ = true;
                    } else {
                        phi_11563_ = (0f >= _e3628_);
                    }
                    let _e3647_ = phi_11563_;
                    let _e3651_ = bitcast<f32>((bitcast<u32>(_e3346_) & 2147483647u));
                    let _e3654_ = bitcast<f32>((bitcast<u32>(_e3351_) & 2147483647u));
                    let _e3657_ = bitcast<f32>((bitcast<u32>(_e3356_) & 2147483647u));
                    let _e3660_ = bitcast<f32>((bitcast<u32>(_e3361_) & 2147483647u));
                    let _e3663_ = bitcast<f32>((bitcast<u32>(_e3388_) & 2147483647u));
                    let _e3666_ = bitcast<f32>((bitcast<u32>(_e3393_) & 2147483647u));
                    let _e3669_ = bitcast<f32>((bitcast<u32>(_e3398_) & 2147483647u));
                    let _e3672_ = bitcast<f32>((bitcast<u32>(_e3403_) & 2147483647u));
                    if (_e3651_ != _e3651_) {
                        phi_11578_ = true;
                    } else {
                        phi_11578_ = (_e3663_ >= _e3651_);
                    }
                    let _e3676_ = phi_11578_;
                    let _e3677_ = select(_e3651_, _e3663_, _e3676_);
                    if (_e3654_ != _e3654_) {
                        phi_11593_ = true;
                    } else {
                        phi_11593_ = (_e3666_ >= _e3654_);
                    }
                    let _e3681_ = phi_11593_;
                    let _e3682_ = select(_e3654_, _e3666_, _e3681_);
                    if (_e3657_ != _e3657_) {
                        phi_11608_ = true;
                    } else {
                        phi_11608_ = (_e3669_ >= _e3657_);
                    }
                    let _e3686_ = phi_11608_;
                    let _e3687_ = select(_e3657_, _e3669_, _e3686_);
                    if (_e3660_ != _e3660_) {
                        phi_11623_ = true;
                    } else {
                        phi_11623_ = (_e3672_ >= _e3660_);
                    }
                    let _e3691_ = phi_11623_;
                    let _e3692_ = select(_e3660_, _e3672_, _e3691_);
                    let _e3709_ = (_e3677_ <= 0f);
                    let _e3710_ = select((_e3525_ * (1.5f - (0.5f * (_e3537_ + _e3541_)))), 0f, _e3709_);
                    let _e3711_ = (_e3682_ <= 0f);
                    let _e3712_ = select((_e3526_ * (1.5f - (0.5f * (_e3538_ + _e3542_)))), 0f, _e3711_);
                    let _e3713_ = (_e3687_ <= 0f);
                    let _e3714_ = select((_e3527_ * (1.5f - (0.5f * (_e3539_ + _e3543_)))), 0f, _e3713_);
                    let _e3715_ = (_e3692_ <= 0f);
                    let _e3716_ = select((_e3528_ * (1.5f - (0.5f * (_e3540_ + _e3544_)))), 0f, _e3715_);
                    let _e3758_ = select(((_e3537_ + (0.33333334f * _e3541_)) - (((_e3537_ * _e3537_) + ((2f * _e3537_) * _e3541_)) + (0.2f * (_e3541_ * _e3541_)))), 0f, _e3709_);
                    let _e3759_ = select(((_e3538_ + (0.33333334f * _e3542_)) - (((_e3538_ * _e3538_) + ((2f * _e3538_) * _e3542_)) + (0.2f * (_e3542_ * _e3542_)))), 0f, _e3711_);
                    let _e3760_ = select(((_e3539_ + (0.33333334f * _e3543_)) - (((_e3539_ * _e3539_) + ((2f * _e3539_) * _e3543_)) + (0.2f * (_e3543_ * _e3543_)))), 0f, _e3713_);
                    let _e3761_ = select(((_e3540_ + (0.33333334f * _e3544_)) - (((_e3540_ * _e3540_) + ((2f * _e3540_) * _e3544_)) + (0.2f * (_e3544_ * _e3544_)))), 0f, _e3715_);
                    if (_e3677_ <= 1f) {
                        if (_e3682_ <= 1f) {
                            let _e3765_ = (_e3687_ <= 1f);
                            if _e3765_ {
                                phi_7263_ = (_e3692_ <= 1f);
                            } else {
                                phi_7263_ = bool();
                            }
                            let _e3768_ = phi_7263_;
                            phi_7265_ = _e3768_;
                            phi_7266_ = select(true, false, _e3765_);
                        } else {
                            phi_7265_ = bool();
                            phi_7266_ = true;
                        }
                        let _e3771_ = phi_7265_;
                        let _e3773_ = phi_7266_;
                        phi_7267_ = _e3771_;
                        phi_7268_ = _e3773_;
                    } else {
                        phi_7267_ = bool();
                        phi_7268_ = true;
                    }
                    let _e3775_ = phi_7267_;
                    let _e3777_ = phi_7268_;
                    if select(_e3775_, false, _e3777_) {
                        phi_7646_ = type_27_(type_22_(array<f32, 4>(_e3710_, _e3712_, _e3714_, _e3716_)), type_22_(array<f32, 4>(_e3758_, _e3759_, _e3760_, _e3761_)), type_22_(_e3404_));
                    } else {
                        if (_e3565_ != _e3565_) {
                            phi_11638_ = true;
                        } else {
                            phi_11638_ = (0.000001f >= _e3565_);
                        }
                        let _e3782_ = phi_11638_;
                        let _e3783_ = select(_e3565_, 0.000001f, _e3782_);
                        if (_e3566_ != _e3566_) {
                            phi_11653_ = true;
                        } else {
                            phi_11653_ = (0.000001f >= _e3566_);
                        }
                        let _e3787_ = phi_11653_;
                        let _e3788_ = select(_e3566_, 0.000001f, _e3787_);
                        if (_e3567_ != _e3567_) {
                            phi_11668_ = true;
                        } else {
                            phi_11668_ = (0.000001f >= _e3567_);
                        }
                        let _e3792_ = phi_11668_;
                        let _e3793_ = select(_e3567_, 0.000001f, _e3792_);
                        if (_e3568_ != _e3568_) {
                            phi_11683_ = true;
                        } else {
                            phi_11683_ = (0.000001f >= _e3568_);
                        }
                        let _e3797_ = phi_11683_;
                        let _e3798_ = select(_e3568_, 0.000001f, _e3797_);
                        let _e3801_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3783_)));
                        let _e3804_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3788_)));
                        let _e3807_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3793_)));
                        let _e3810_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3798_)));
                        let _e3820_ = (_e3801_ + (_e3801_ * (1f - (_e3783_ * _e3801_))));
                        let _e3822_ = (_e3804_ + (_e3804_ * (1f - (_e3788_ * _e3804_))));
                        let _e3824_ = (_e3807_ + (_e3807_ * (1f - (_e3793_ * _e3807_))));
                        let _e3826_ = (_e3810_ + (_e3810_ * (1f - (_e3798_ * _e3810_))));
                        let _e3836_ = (_e3820_ + (_e3820_ * (1f - (_e3783_ * _e3820_))));
                        let _e3838_ = (_e3822_ + (_e3822_ * (1f - (_e3788_ * _e3822_))));
                        let _e3840_ = (_e3824_ + (_e3824_ * (1f - (_e3793_ * _e3824_))));
                        let _e3842_ = (_e3826_ + (_e3826_ * (1f - (_e3798_ * _e3826_))));
                        let _e3895_ = (select((select(0f, 1f, (-1f <= _e3409_)) * select(0f, 1f, (_e3428_ <= 1f))), 1f, (_e3565_ <= 0.000001f)) <= 0f);
                        let _e3897_ = (select((select(0f, 1f, (-1f <= _e3414_)) * select(0f, 1f, (_e3432_ <= 1f))), 1f, (_e3566_ <= 0.000001f)) <= 0f);
                        let _e3899_ = (select((select(0f, 1f, (-1f <= _e3419_)) * select(0f, 1f, (_e3436_ <= 1f))), 1f, (_e3567_ <= 0.000001f)) <= 0f);
                        let _e3901_ = (select((select(0f, 1f, (-1f <= _e3424_)) * select(0f, 1f, (_e3440_ <= 1f))), 1f, (_e3568_ <= 0.000001f)) <= 0f);
                        let _e3907_ = (1f <= _e3409_);
                        let _e3909_ = (1f <= _e3414_);
                        let _e3911_ = (1f <= _e3419_);
                        let _e3913_ = (1f <= _e3424_);
                        let _e3915_ = (_e3428_ <= -1f);
                        let _e3917_ = (_e3432_ <= -1f);
                        let _e3919_ = (_e3436_ <= -1f);
                        let _e3921_ = (_e3440_ <= -1f);
                        phi_7646_ = type_27_(type_22_(array<f32, 4>(select(select(select(_e3710_, (((select(_e3625_, 0f, _e3632_) - select(_e3585_, 0f, _e3592_)) + (_e3549_ * _e3710_)) * _e3836_), _e3895_), 1f, _e3907_), -1f, _e3915_), select(select(select(_e3712_, (((select(_e3626_, 0f, _e3637_) - select(_e3586_, 0f, _e3597_)) + (_e3554_ * _e3712_)) * _e3838_), _e3897_), 1f, _e3909_), -1f, _e3917_), select(select(select(_e3714_, (((select(_e3627_, 0f, _e3642_) - select(_e3587_, 0f, _e3602_)) + (_e3559_ * _e3714_)) * _e3840_), _e3899_), 1f, _e3911_), -1f, _e3919_), select(select(select(_e3716_, (((select(_e3628_, 0f, _e3647_) - select(_e3588_, 0f, _e3607_)) + (_e3564_ * _e3716_)) * _e3842_), _e3901_), 1f, _e3913_), -1f, _e3921_))), type_22_(array<f32, 4>(select(select(select(_e3758_, ((_e3549_ * _e3758_) * _e3836_), _e3895_), 0f, _e3907_), 0f, _e3915_), select(select(select(_e3759_, ((_e3554_ * _e3759_) * _e3838_), _e3897_), 0f, _e3909_), 0f, _e3917_), select(select(select(_e3760_, ((_e3559_ * _e3760_) * _e3840_), _e3899_), 0f, _e3911_), 0f, _e3919_), select(select(select(_e3761_, ((_e3564_ * _e3761_) * _e3842_), _e3901_), 0f, _e3913_), 0f, _e3921_))), type_22_(_e3404_));
                    }
                    let _e3942_ = phi_7646_;
                    if _e158_ {
                        phi_7655_ = (_e201_ + _e3942_.member.member[0]);
                        phi_7656_ = (_e203_ + _e3317_.member.member[0]);
                    } else {
                        phi_7655_ = _e201_;
                        phi_7656_ = _e203_;
                    }
                    let _e3950_ = phi_7655_;
                    let _e3952_ = phi_7656_;
                    workgroupBarrier();
                    phi_12445_ = _e2621_;
                    phi_306_ = _e3950_;
                    phi_308_ = _e3952_;
                } else {
                    phi_12445_ = _e197_;
                    phi_306_ = f32();
                    phi_308_ = f32();
                }
                let _e3954_ = phi_12445_;
                let _e3956_ = phi_306_;
                let _e3958_ = phi_308_;
                phi_12004_ = _e3954_;
                phi_302_ = _e216_;
                phi_305_ = _e3956_;
                phi_307_ = _e3958_;
                phi_12444_ = _e3954_;
                if !(_e222_) {
                    break;
                }
            }
            let _e3961_ = phi_12444_;
            if _e3961_ {
                break;
            }
            if _e158_ {
                if (_e117_ == 0u) {
                    phi_7670_ = 0u;
                } else {
                    let _e3963_ = (_e117_ - 1u);
                    phi_7670_ = select(_e3963_, _e160_, (_e160_ < _e3963_));
                }
                let _e3967_ = phi_7670_;
                if (_e3967_ < _e117_) {
                } else {
                    break;
                }
                let _e3972_ = target_audio.member[_e3967_].x;
                let _e3976_ = target_audio.member[_e3967_].y;
                let _e3977_ = bitcast<u32>(_e3972_);
                let _e3978_ = (_e3977_ & 2139095040u);
                if (_e3978_ == 2139095040u) {
                    phi_7693_ = true;
                } else {
                    phi_7693_ = (_e3978_ == 0u);
                }
                let _e3982_ = phi_7693_;
                let _e3985_ = bitcast<u32>(_e3976_);
                let _e3986_ = (_e3985_ & 2139095040u);
                if (_e3986_ == 2139095040u) {
                    phi_7709_ = true;
                } else {
                    phi_7709_ = (_e3986_ == 0u);
                }
                let _e3990_ = phi_7709_;
                phi_7717_ = vec2<f32>(bitcast<f32>(select(_e3977_, 0u, _e3982_)), bitcast<f32>(select(_e3985_, 0u, _e3990_)));
            } else {
                phi_7717_ = vec2<f32>(0f, 0f);
            }
            let _e3995_ = phi_7717_;
            if _e158_ {
                let _e3998_ = local;
                phi_7723_ = (_e3998_ - _e3995_.x);
            } else {
                phi_7723_ = 0f;
            }
            let _e4001_ = phi_7723_;
            if _e158_ {
                let _e4004_ = local_1_;
                phi_7729_ = (_e4004_ - _e3995_.y);
            } else {
                phi_7729_ = 0f;
            }
            let _e4007_ = phi_7729_;
            if _e158_ {
                let _e4011_ = bitcast<u32>(((_e4001_ * _e4001_) + (_e4007_ * _e4007_)));
                let _e4012_ = (_e4011_ & 2139095040u);
                if (_e4012_ == 2139095040u) {
                    phi_7746_ = true;
                } else {
                    phi_7746_ = (_e4012_ == 0u);
                }
                let _e4016_ = phi_7746_;
                phi_7752_ = bitcast<f32>(select(_e4011_, 0u, _e4016_));
            } else {
                phi_7752_ = 0f;
            }
            let _e4020_ = phi_7752_;
            if _e158_ {
                let _e4026_ = bitcast<u32>(((_e3995_.x * _e3995_.x) + (_e3995_.y * _e3995_.y)));
                let _e4027_ = (_e4026_ & 2139095040u);
                if (_e4027_ == 2139095040u) {
                    phi_7773_ = true;
                } else {
                    phi_7773_ = (_e4027_ == 0u);
                }
                let _e4031_ = phi_7773_;
                phi_7779_ = bitcast<f32>(select(_e4026_, 0u, _e4031_));
            } else {
                phi_7779_ = 0f;
            }
            let _e4035_ = phi_7779_;
            let _e4036_ = bitcast<u32>(_e4020_);
            let _e4037_ = (_e4036_ & 2139095040u);
            if (_e4037_ == 2139095040u) {
                phi_7790_ = true;
            } else {
                phi_7790_ = (_e4037_ == 0u);
            }
            let _e4041_ = phi_7790_;
            let _e4044_ = bitcast<u32>(_e4035_);
            let _e4045_ = (_e4044_ & 2139095040u);
            if (_e4045_ == 2139095040u) {
                phi_7806_ = true;
            } else {
                phi_7806_ = (_e4045_ == 0u);
            }
            let _e4049_ = phi_7806_;
            let _e4053_ = select(255u, _e113_, (_e113_ < 255u));
            let _e4054_ = (_e4053_ < 256u);
            if _e4054_ {
            } else {
                break;
            }
            err_scratch[_e4053_] = bitcast<f32>(select(_e4036_, 0u, _e4041_));
            if _e4054_ {
            } else {
                break;
            }
            tgt_scratch[_e4053_] = bitcast<f32>(select(_e4044_, 0u, _e4049_));
            workgroupBarrier();
            phi_7836_ = 128u;
            loop {
                let _e4058_ = phi_7836_;
                let _e4059_ = (_e4058_ > 0u);
                if _e4059_ {
                    if (_e113_ < _e4058_) {
                        let _e4061_ = (_e113_ + _e4058_);
                        let _e4062_ = (_e4061_ < 256u);
                        if _e4062_ {
                        } else {
                            phi_12852_ = true;
                            break;
                        }
                        let _e4064_ = err_scratch[_e4061_];
                        let _e4065_ = (_e113_ < 256u);
                        if _e4065_ {
                        } else {
                            phi_12852_ = true;
                            break;
                        }
                        let _e4067_ = err_scratch[_e113_];
                        err_scratch[_e113_] = (_e4067_ + _e4064_);
                        if _e4062_ {
                        } else {
                            phi_12852_ = true;
                            break;
                        }
                        let _e4070_ = tgt_scratch[_e4061_];
                        if _e4065_ {
                        } else {
                            phi_12852_ = true;
                            break;
                        }
                        let _e4072_ = tgt_scratch[_e113_];
                        tgt_scratch[_e113_] = (_e4072_ + _e4070_);
                    }
                    workgroupBarrier();
                    phi_7837_ = (_e4058_ >> bitcast<u32>(1i));
                } else {
                    phi_7837_ = u32();
                }
                let _e4077_ = phi_7837_;
                phi_7836_ = _e4077_;
                phi_12852_ = _e3961_;
                if !(_e4059_) {
                    break;
                }
            }
            let _e4080_ = phi_12852_;
            if _e4080_ {
                break;
            }
            let _e4082_ = err_scratch[0];
            let _e4083_ = bitcast<u32>(_e4082_);
            let _e4084_ = (_e4083_ & 2139095040u);
            if (_e4084_ == 2139095040u) {
                phi_7897_ = true;
            } else {
                phi_7897_ = (_e4084_ == 0u);
            }
            let _e4088_ = phi_7897_;
            let _e4092_ = tgt_scratch[0];
            let _e4093_ = bitcast<u32>(_e4092_);
            let _e4094_ = (_e4093_ & 2139095040u);
            if (_e4094_ == 2139095040u) {
                phi_7915_ = true;
            } else {
                phi_7915_ = (_e4094_ == 0u);
            }
            let _e4098_ = phi_7915_;
            workgroupBarrier();
            if (_e113_ == 0u) {
                if (_e112_.x < _e129_) {
                    if (_e112_.y < _e149_) {
                        if (_e112_.z < select(1u, _e140_, (1u < _e140_))) {
                            let _e4107_ = uniforms.member.member_5_;
                            if ((bitcast<u32>(_e4107_) & 2139095040u) != 2139095040u) {
                                let _e4111_ = (_e4107_ > 0.000000000001f);
                                phi_7955_ = select(f32(), _e4107_, _e4111_);
                                phi_7956_ = select(true, false, _e4111_);
                            } else {
                                phi_7955_ = f32();
                                phi_7956_ = true;
                            }
                            let _e4115_ = phi_7955_;
                            let _e4117_ = phi_7956_;
                            let _e4119_ = (bitcast<f32>(select(_e4093_, 0u, _e4098_)) + select(_e4115_, 0.000000000001f, _e4117_));
                            if ((bitcast<u32>(_e4119_) & 2139095040u) != 2139095040u) {
                                let _e4123_ = (_e4119_ <= 0f);
                                if _e4123_ {
                                    phi_8054_ = f32();
                                } else {
                                    if (_e4119_ < 0.00001f) {
                                        let _e4155_ = (select(0.000000000000000000000001f, _e4119_, (_e4119_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                        if (_e4155_ != _e4155_) {
                                            phi_11758_ = true;
                                        } else {
                                            phi_11758_ = (0.000001f >= _e4155_);
                                        }
                                        let _e4159_ = phi_11758_;
                                        let _e4160_ = select(_e4155_, 0.000001f, _e4159_);
                                        let _e4163_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4160_)));
                                        let _e4167_ = (_e4163_ + (_e4163_ * (1f - (_e4160_ * _e4163_))));
                                        let _e4171_ = (_e4167_ + (_e4167_ * (1f - (_e4160_ * _e4167_))));
                                        let _e4177_ = bitcast<u32>(((_e4171_ + (_e4171_ * (1f - (_e4160_ * _e4171_)))) * 18446744000000000000f));
                                        let _e4178_ = (_e4177_ & 2139095040u);
                                        if (_e4178_ == 2139095040u) {
                                            phi_8012_ = true;
                                        } else {
                                            phi_8012_ = (_e4178_ == 0u);
                                        }
                                        let _e4182_ = phi_8012_;
                                        phi_8053_ = bitcast<f32>(select(_e4177_, 0u, _e4182_));
                                    } else {
                                        if (_e4119_ != _e4119_) {
                                            phi_11698_ = true;
                                        } else {
                                            phi_11698_ = (0.000001f >= _e4119_);
                                        }
                                        let _e4128_ = phi_11698_;
                                        let _e4129_ = select(_e4119_, 0.000001f, _e4128_);
                                        let _e4132_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4129_)));
                                        let _e4136_ = (_e4132_ + (_e4132_ * (1f - (_e4129_ * _e4132_))));
                                        let _e4140_ = (_e4136_ + (_e4136_ * (1f - (_e4129_ * _e4136_))));
                                        let _e4145_ = bitcast<u32>((_e4140_ + (_e4140_ * (1f - (_e4129_ * _e4140_)))));
                                        let _e4146_ = (_e4145_ & 2139095040u);
                                        if (_e4146_ == 2139095040u) {
                                            phi_8047_ = true;
                                        } else {
                                            phi_8047_ = (_e4146_ == 0u);
                                        }
                                        let _e4150_ = phi_8047_;
                                        phi_8053_ = bitcast<f32>(select(_e4145_, 0u, _e4150_));
                                    }
                                    let _e4186_ = phi_8053_;
                                    phi_8054_ = _e4186_;
                                }
                                let _e4188_ = phi_8054_;
                                phi_8055_ = _e4188_;
                                phi_8056_ = _e4123_;
                            } else {
                                phi_8055_ = f32();
                                phi_8056_ = true;
                            }
                            let _e4190_ = phi_8055_;
                            let _e4192_ = phi_8056_;
                            let _e4195_ = bitcast<u32>((bitcast<f32>(select(_e4083_, 0u, _e4088_)) * select(_e4190_, 0f, _e4192_)));
                            let _e4196_ = (_e4195_ & 2139095040u);
                            if (_e4196_ == 2139095040u) {
                                phi_8072_ = true;
                            } else {
                                phi_8072_ = (_e4196_ == 0u);
                            }
                            let _e4200_ = phi_8072_;
                            let _e4206_ = ((((_e112_.z * _e129_) + _e112_.x) * _e149_) + _e112_.y);
                            if (_e121_ == 0u) {
                                phi_8092_ = 0u;
                            } else {
                                let _e4208_ = (_e121_ - 1u);
                                phi_8092_ = select(_e4208_, _e4206_, (_e4206_ < _e4208_));
                            }
                            let _e4212_ = phi_8092_;
                            if (_e4212_ < _e121_) {
                            } else {
                                break;
                            }
                            seed_losses.member[_e4212_] = bitcast<f32>(select(_e4195_, 0u, _e4200_));
                        }
                    }
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn prescreen_population(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    function_();
    return;
}

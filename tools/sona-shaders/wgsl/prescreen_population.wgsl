struct type_6_ {
    member: array<f32>,
}

struct type_15_ {
    member: u32,
    member_1_: u32,
}

struct type_21_ {
    member: f32,
    member_1_: f32,
}

struct type_22_ {
    member: f32,
    member_1_: f32,
    member_2_: f32,
    member_3_: f32,
    member_4_: f32,
    member_5_: f32,
}

struct type_23_ {
    member: array<f32, 4>,
}

struct type_25_ {
    member: array<vec2<f32>>,
}

struct type_27_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: f32,
    member_5_: f32,
    member_6_: u32,
    member_7_: u32,
}

struct type_28_ {
    member: type_27_,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<workgroup> scan_scratch: array<f32, 1536>;
@group(0) @binding(2) 
var<storage> target_audio: type_25_;
@group(0) @binding(0) 
var<uniform> uniforms: type_28_;
@group(0) @binding(1) 
var<storage> input_audio: type_25_;
@group(0) @binding(3) 
var<storage> seed_params: type_6_;
@group(0) @binding(4) 
var<storage, read_write> seed_losses: type_6_;
var<workgroup> err_scratch: array<f32, 256>;
var<workgroup> tgt_scratch: array<f32, 256>;

fn function_() {
    var phi_219_: u32;
    var phi_226_: u32;
    var phi_249_: u32;
    var phi_272_: bool;
    var phi_288_: bool;
    var phi_296_: vec2<f32>;
    var phi_10383_: bool;
    var phi_299_: type_15_;
    var phi_302_: f32;
    var phi_304_: f32;
    var phi_300_: type_15_;
    var phi_327_: type_15_;
    var phi_345_: u32;
    var phi_356_: u32;
    var phi_368_: u32;
    var phi_387_: u32;
    var phi_420_: f32;
    var phi_426_: f32;
    var phi_7227_: bool;
    var phi_597_: bool;
    var phi_603_: f32;
    var phi_604_: f32;
    var phi_523_: f32;
    var phi_525_: f32;
    var phi_526_: bool;
    var phi_546_: f32;
    var phi_605_: f32;
    var phi_606_: f32;
    var phi_7287_: bool;
    var phi_692_: bool;
    var phi_7347_: bool;
    var phi_657_: bool;
    var phi_698_: f32;
    var phi_699_: f32;
    var phi_700_: f32;
    var phi_701_: bool;
    var phi_717_: u32;
    var phi_735_: bool;
    var phi_752_: u32;
    var phi_770_: bool;
    var phi_787_: u32;
    var phi_805_: bool;
    var phi_822_: u32;
    var phi_840_: bool;
    var phi_857_: u32;
    var phi_875_: bool;
    var phi_892_: u32;
    var phi_910_: bool;
    var phi_7407_: bool;
    var phi_7467_: bool;
    var phi_1491_: bool;
    var phi_1497_: f32;
    var phi_1498_: f32;
    var phi_1417_: f32;
    var phi_1419_: f32;
    var phi_1420_: bool;
    var phi_1499_: f32;
    var phi_1511_: u32;
    var phi_1529_: bool;
    var phi_1546_: u32;
    var phi_1564_: bool;
    var phi_1581_: u32;
    var phi_1599_: bool;
    var phi_1616_: u32;
    var phi_1634_: bool;
    var phi_1651_: u32;
    var phi_1775_: bool;
    var phi_1781_: f32;
    var phi_1782_: f32;
    var phi_1701_: f32;
    var phi_1703_: f32;
    var phi_1704_: bool;
    var phi_1724_: f32;
    var phi_1783_: f32;
    var phi_1795_: u32;
    var phi_1919_: bool;
    var phi_1925_: f32;
    var phi_1926_: f32;
    var phi_1845_: f32;
    var phi_1847_: f32;
    var phi_1848_: bool;
    var phi_1868_: f32;
    var phi_1927_: f32;
    var phi_1939_: u32;
    var phi_2063_: bool;
    var phi_2069_: f32;
    var phi_2070_: f32;
    var phi_1989_: f32;
    var phi_1991_: f32;
    var phi_1992_: bool;
    var phi_2012_: f32;
    var phi_2071_: f32;
    var phi_2083_: u32;
    var phi_2207_: bool;
    var phi_2213_: f32;
    var phi_2214_: f32;
    var phi_2133_: f32;
    var phi_2135_: f32;
    var phi_2136_: bool;
    var phi_2156_: f32;
    var phi_2215_: f32;
    var phi_2237_: type_22_;
    var phi_2297_: u32;
    var phi_2298_: u32;
    var phi_10187_: bool;
    var phi_2560_: u32;
    var phi_2561_: u32;
    var phi_10488_: bool;
    var phi_3103_: bool;
    var phi_3119_: bool;
    var phi_3127_: type_21_;
    var phi_3150_: type_22_;
    var phi_3210_: u32;
    var phi_3211_: u32;
    var phi_10503_: bool;
    var phi_3473_: u32;
    var phi_3474_: u32;
    var phi_10523_: bool;
    var phi_4016_: bool;
    var phi_4032_: bool;
    var phi_4040_: type_21_;
    var phi_4105_: type_15_;
    var phi_4108_: type_23_;
    var phi_4110_: type_23_;
    var phi_4106_: type_15_;
    var phi_4133_: type_15_;
    var phi_4186_: bool;
    var phi_4202_: bool;
    var phi_4218_: bool;
    var phi_4234_: bool;
    var phi_7527_: bool;
    var phi_7542_: bool;
    var phi_7557_: bool;
    var phi_7572_: bool;
    var phi_7587_: bool;
    var phi_7602_: bool;
    var phi_7617_: bool;
    var phi_7632_: bool;
    var phi_7647_: bool;
    var phi_7662_: bool;
    var phi_7677_: bool;
    var phi_7692_: bool;
    var phi_7707_: bool;
    var phi_7722_: bool;
    var phi_7737_: bool;
    var phi_7752_: bool;
    var phi_7767_: bool;
    var phi_7782_: bool;
    var phi_7797_: bool;
    var phi_7812_: bool;
    var phi_7827_: bool;
    var phi_7842_: bool;
    var phi_7857_: bool;
    var phi_7872_: bool;
    var phi_7887_: bool;
    var phi_7902_: bool;
    var phi_7917_: bool;
    var phi_7932_: bool;
    var phi_7947_: bool;
    var phi_7962_: bool;
    var phi_7977_: bool;
    var phi_7992_: bool;
    var phi_4641_: bool;
    var phi_4657_: bool;
    var phi_4673_: bool;
    var phi_4689_: bool;
    var phi_4711_: bool;
    var phi_4727_: bool;
    var phi_4743_: bool;
    var phi_4759_: bool;
    var phi_8007_: bool;
    var phi_8022_: bool;
    var phi_8037_: bool;
    var phi_8052_: bool;
    var phi_8067_: bool;
    var phi_8082_: bool;
    var phi_8097_: bool;
    var phi_8112_: bool;
    var phi_8127_: bool;
    var phi_8142_: bool;
    var phi_8157_: bool;
    var phi_8172_: bool;
    var phi_8187_: bool;
    var phi_8202_: bool;
    var phi_8217_: bool;
    var phi_8232_: bool;
    var phi_8247_: bool;
    var phi_8262_: bool;
    var phi_8277_: bool;
    var phi_8292_: bool;
    var phi_8307_: bool;
    var phi_8322_: bool;
    var phi_8337_: bool;
    var phi_8352_: bool;
    var phi_8367_: bool;
    var phi_8382_: bool;
    var phi_8397_: bool;
    var phi_8412_: bool;
    var phi_8427_: bool;
    var phi_8442_: bool;
    var phi_8457_: bool;
    var phi_8472_: bool;
    var phi_5166_: bool;
    var phi_5182_: bool;
    var phi_5198_: bool;
    var phi_5214_: bool;
    var phi_4109_: type_23_;
    var phi_4111_: type_23_;
    var phi_10538_: bool;
    var phi_10966_: bool;
    var phi_5302_: f32;
    var phi_5314_: f32;
    var phi_8487_: bool;
    var phi_8502_: bool;
    var phi_8517_: bool;
    var phi_8577_: bool;
    var phi_5370_: f32;
    var phi_8637_: bool;
    var phi_8697_: bool;
    var phi_5420_: f32;
    var phi_5449_: f32;
    var phi_5451_: f32;
    var phi_5452_: bool;
    var phi_5473_: f32;
    var phi_5475_: f32;
    var phi_5476_: bool;
    var phi_8757_: bool;
    var phi_8817_: bool;
    var phi_8877_: bool;
    var phi_8937_: bool;
    var phi_8997_: bool;
    var phi_9057_: bool;
    var phi_9117_: bool;
    var phi_9177_: bool;
    var phi_9237_: bool;
    var phi_9417_: bool;
    var phi_9477_: bool;
    var phi_9537_: bool;
    var phi_9597_: bool;
    var phi_9657_: bool;
    var phi_9717_: bool;
    var phi_9777_: bool;
    var phi_9837_: bool;
    var phi_9897_: bool;
    var phi_6754_: f32;
    var phi_6755_: f32;
    var phi_10744_: bool;
    var phi_303_: f32;
    var phi_305_: f32;
    var phi_10743_: bool;
    var phi_6769_: u32;
    var phi_6792_: bool;
    var phi_6808_: bool;
    var phi_6816_: vec2<f32>;
    var local: f32;
    var phi_6822_: f32;
    var local_1_: f32;
    var phi_6828_: f32;
    var phi_6845_: bool;
    var phi_6851_: f32;
    var phi_6872_: bool;
    var phi_6878_: f32;
    var phi_6889_: bool;
    var phi_6905_: bool;
    var phi_6935_: u32;
    var phi_6936_: u32;
    var phi_10979_: bool;
    var phi_6996_: bool;
    var phi_7014_: bool;
    var phi_7054_: f32;
    var phi_7055_: bool;
    var phi_10077_: bool;
    var phi_7146_: bool;
    var phi_10137_: bool;
    var phi_7111_: bool;
    var phi_7152_: f32;
    var phi_7153_: f32;
    var phi_7154_: f32;
    var phi_7155_: bool;
    var phi_7171_: bool;
    var phi_7191_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e109_ = wg_id_1_;
            let _e110_ = lid_1_;
            let _e112_ = arrayLength((&input_audio.member));
            let _e114_ = arrayLength((&target_audio.member));
            let _e116_ = arrayLength((&seed_params.member));
            let _e118_ = arrayLength((&seed_losses.member));
            let _e124_ = uniforms.member.member;
            let _e126_ = select(1u, _e124_, (1u < _e124_));
            let _e129_ = uniforms.member.member_1_;
            let _e132_ = uniforms.member.member_2_;
            let _e134_ = select(1u, _e132_, (1u < _e132_));
            let _e137_ = uniforms.member.member_3_;
            if (_e129_ == 0u) {
                phi_219_ = 1u;
            } else {
                phi_219_ = (((_e129_ - 1u) >> bitcast<u32>(8i)) + 1u);
            }
            let _e146_ = phi_219_;
            let _e147_ = (_e109_.y * 256u);
            if (_e129_ > _e147_) {
                phi_226_ = (_e129_ - _e147_);
            } else {
                phi_226_ = 0u;
            }
            let _e151_ = phi_226_;
            let _e154_ = (_e147_ + _e110_);
            let _e155_ = (_e154_ < _e129_);
            let _e157_ = ((_e109_.z * _e129_) + _e154_);
            if _e155_ {
                if (_e112_ == 0u) {
                    phi_249_ = 0u;
                } else {
                    let _e159_ = (_e112_ - 1u);
                    phi_249_ = select(_e159_, _e157_, (_e157_ < _e159_));
                }
                let _e163_ = phi_249_;
                if (_e163_ < _e112_) {
                } else {
                    break;
                }
                let _e168_ = input_audio.member[_e163_].x;
                let _e172_ = input_audio.member[_e163_].y;
                let _e173_ = bitcast<u32>(_e168_);
                let _e174_ = (_e173_ & 2139095040u);
                if (_e174_ == 2139095040u) {
                    phi_272_ = true;
                } else {
                    phi_272_ = (_e174_ == 0u);
                }
                let _e178_ = phi_272_;
                let _e181_ = bitcast<u32>(_e172_);
                let _e182_ = (_e181_ & 2139095040u);
                if (_e182_ == 2139095040u) {
                    phi_288_ = true;
                } else {
                    phi_288_ = (_e182_ == 0u);
                }
                let _e186_ = phi_288_;
                phi_296_ = vec2<f32>(bitcast<f32>(select(_e173_, 0u, _e178_)), bitcast<f32>(select(_e181_, 0u, _e186_)));
            } else {
                phi_296_ = vec2<f32>(0f, 0f);
            }
            let _e191_ = phi_296_;
            phi_10383_ = false;
            phi_299_ = type_15_(0u, _e134_);
            phi_302_ = 0f;
            phi_304_ = 0f;
            loop {
                let _e194_ = phi_10383_;
                let _e196_ = phi_299_;
                let _e198_ = phi_302_;
                let _e200_ = phi_304_;
                local = _e200_;
                local_1_ = _e198_;
                if (_e196_.member < _e196_.member_1_) {
                    phi_300_ = type_15_((_e196_.member + 1u), _e196_.member_1_);
                    phi_327_ = type_15_(1u, _e196_.member);
                } else {
                    phi_300_ = _e196_;
                    phi_327_ = type_15_(0u, u32());
                }
                let _e213_ = phi_300_;
                let _e215_ = phi_327_;
                let _e219_ = (bitcast<i32>(_e215_.member) != 0i);
                if _e219_ {
                    if (_e126_ == 0u) {
                        phi_345_ = 0u;
                    } else {
                        let _e221_ = (_e126_ - 1u);
                        phi_345_ = select(_e221_, _e109_.x, (_e109_.x < _e221_));
                    }
                    let _e225_ = phi_345_;
                    if (_e116_ >= ((_e126_ * _e134_) * 16u)) {
                        phi_356_ = (((_e225_ * _e134_) + _e215_.member_1_) * 16u);
                    } else {
                        phi_356_ = (_e225_ * 16u);
                    }
                    let _e234_ = phi_356_;
                    let _e235_ = (_e116_ == 0u);
                    if _e235_ {
                        phi_368_ = 0u;
                    } else {
                        let _e236_ = (_e116_ - 1u);
                        phi_368_ = select(_e236_, _e234_, (_e234_ < _e236_));
                    }
                    let _e240_ = phi_368_;
                    if (_e240_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e244_ = seed_params.member[_e240_];
                    let _e245_ = (_e234_ + 1u);
                    if _e235_ {
                        phi_387_ = 0u;
                    } else {
                        let _e246_ = (_e116_ - 1u);
                        phi_387_ = select(_e246_, _e245_, (_e245_ < _e246_));
                    }
                    let _e250_ = phi_387_;
                    if (_e250_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e254_ = seed_params.member[_e250_];
                    let _e258_ = select(0.001f, _e244_, ((bitcast<u32>(_e244_) & 2139095040u) != 2139095040u));
                    if (_e258_ != _e258_) {
                        phi_420_ = 0.00001f;
                    } else {
                        let _e261_ = select(_e258_, 0.00001f, (_e258_ < 0.00001f));
                        phi_420_ = select(_e261_, 0.495f, (_e261_ > 0.495f));
                    }
                    let _e265_ = phi_420_;
                    let _e266_ = (_e265_ <= 0.25f);
                    if _e266_ {
                        phi_426_ = _e265_;
                    } else {
                        phi_426_ = (0.5f - _e265_);
                    }
                    let _e269_ = phi_426_;
                    let _e270_ = (3.1415927f * _e269_);
                    let _e271_ = (_e270_ * _e270_);
                    let _e282_ = (_e270_ * (1f + (_e271_ * (-0.16666667f + (_e271_ * (0.008333334f + (_e271_ * (-0.0001984127f + (_e271_ * (0.0000027557319f + (_e271_ * -0.000000025052108f)))))))))));
                    let _e292_ = (1f + (_e271_ * (-0.5f + (_e271_ * (0.041666668f + (_e271_ * (-0.0013888889f + (_e271_ * (0.000024801588f + (_e271_ * -0.0000002755732f))))))))));
                    let _e294_ = select(_e282_, _e292_, _e266_);
                    if (_e294_ != _e294_) {
                        phi_7227_ = true;
                    } else {
                        phi_7227_ = (0.000001f >= _e294_);
                    }
                    let _e298_ = phi_7227_;
                    let _e299_ = select(_e294_, 0.000001f, _e298_);
                    let _e302_ = bitcast<f32>((2129859011u - bitcast<u32>(_e299_)));
                    let _e306_ = (_e302_ + (_e302_ * (1f - (_e299_ * _e302_))));
                    let _e310_ = (_e306_ + (_e306_ * (1f - (_e299_ * _e306_))));
                    let _e315_ = (select(_e292_, _e282_, _e266_) * (_e310_ + (_e310_ * (1f - (_e299_ * _e310_)))));
                    let _e316_ = bitcast<u32>(_e254_);
                    let _e317_ = (_e316_ & 2139095040u);
                    if (_e317_ != 2139095040u) {
                        let _e319_ = (_e317_ == 2139095040u);
                        if _e319_ {
                            if ((_e316_ & 2147483647u) > 2139095040u) {
                                phi_525_ = f32();
                                phi_526_ = true;
                            } else {
                                let _e334_ = ((_e316_ & 2147483648u) == 0u);
                                if _e334_ {
                                    phi_523_ = 64f;
                                } else {
                                    phi_523_ = f32();
                                }
                                let _e336_ = phi_523_;
                                phi_525_ = _e336_;
                                phi_526_ = select(true, false, _e334_);
                            }
                            let _e339_ = phi_525_;
                            let _e341_ = phi_526_;
                            if _e341_ {
                                phi_546_ = 0.05f;
                            } else {
                                phi_546_ = _e339_;
                            }
                            let _e343_ = phi_546_;
                            phi_605_ = _e343_;
                        } else {
                            if (_e254_ < 0.05f) {
                                phi_604_ = 0.05f;
                            } else {
                                if (_e254_ > 64f) {
                                    phi_603_ = 64f;
                                } else {
                                    if _e319_ {
                                        phi_597_ = true;
                                    } else {
                                        phi_597_ = (_e317_ == 0u);
                                    }
                                    let _e324_ = phi_597_;
                                    phi_603_ = bitcast<f32>(select(_e316_, 0u, _e324_));
                                }
                                let _e328_ = phi_603_;
                                phi_604_ = _e328_;
                            }
                            let _e330_ = phi_604_;
                            phi_605_ = _e330_;
                        }
                        let _e345_ = phi_605_;
                        phi_606_ = _e345_;
                    } else {
                        phi_606_ = 0.70710677f;
                    }
                    let _e347_ = phi_606_;
                    if ((bitcast<u32>(_e347_) & 2139095040u) != 2139095040u) {
                        let _e351_ = (_e347_ <= 0f);
                        if _e351_ {
                            phi_699_ = f32();
                        } else {
                            if (_e347_ < 0.00001f) {
                                let _e383_ = (select(0.000000000000000000000001f, _e347_, (_e347_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                if (_e383_ != _e383_) {
                                    phi_7347_ = true;
                                } else {
                                    phi_7347_ = (0.000001f >= _e383_);
                                }
                                let _e387_ = phi_7347_;
                                let _e388_ = select(_e383_, 0.000001f, _e387_);
                                let _e391_ = bitcast<f32>((2129859011u - bitcast<u32>(_e388_)));
                                let _e395_ = (_e391_ + (_e391_ * (1f - (_e388_ * _e391_))));
                                let _e399_ = (_e395_ + (_e395_ * (1f - (_e388_ * _e395_))));
                                let _e405_ = bitcast<u32>(((_e399_ + (_e399_ * (1f - (_e388_ * _e399_)))) * 18446744000000000000f));
                                let _e406_ = (_e405_ & 2139095040u);
                                if (_e406_ == 2139095040u) {
                                    phi_657_ = true;
                                } else {
                                    phi_657_ = (_e406_ == 0u);
                                }
                                let _e410_ = phi_657_;
                                phi_698_ = bitcast<f32>(select(_e405_, 0u, _e410_));
                            } else {
                                if (_e347_ != _e347_) {
                                    phi_7287_ = true;
                                } else {
                                    phi_7287_ = (0.000001f >= _e347_);
                                }
                                let _e356_ = phi_7287_;
                                let _e357_ = select(_e347_, 0.000001f, _e356_);
                                let _e360_ = bitcast<f32>((2129859011u - bitcast<u32>(_e357_)));
                                let _e364_ = (_e360_ + (_e360_ * (1f - (_e357_ * _e360_))));
                                let _e368_ = (_e364_ + (_e364_ * (1f - (_e357_ * _e364_))));
                                let _e373_ = bitcast<u32>((_e368_ + (_e368_ * (1f - (_e357_ * _e368_)))));
                                let _e374_ = (_e373_ & 2139095040u);
                                if (_e374_ == 2139095040u) {
                                    phi_692_ = true;
                                } else {
                                    phi_692_ = (_e374_ == 0u);
                                }
                                let _e378_ = phi_692_;
                                phi_698_ = bitcast<f32>(select(_e373_, 0u, _e378_));
                            }
                            let _e414_ = phi_698_;
                            phi_699_ = _e414_;
                        }
                        let _e416_ = phi_699_;
                        phi_700_ = _e416_;
                        phi_701_ = _e351_;
                    } else {
                        phi_700_ = f32();
                        phi_701_ = true;
                    }
                    let _e418_ = phi_700_;
                    let _e420_ = phi_701_;
                    let _e421_ = select(_e418_, 0f, _e420_);
                    let _e422_ = (_e234_ + 6u);
                    if _e235_ {
                        phi_717_ = 0u;
                    } else {
                        let _e423_ = (_e116_ - 1u);
                        phi_717_ = select(_e423_, _e422_, (_e422_ < _e423_));
                    }
                    let _e427_ = phi_717_;
                    if (_e427_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e431_ = seed_params.member[_e427_];
                    let _e432_ = bitcast<u32>(_e431_);
                    let _e433_ = (_e432_ & 2139095040u);
                    if (_e433_ == 2139095040u) {
                        phi_735_ = true;
                    } else {
                        phi_735_ = (_e433_ == 0u);
                    }
                    let _e437_ = phi_735_;
                    let _e439_ = bitcast<f32>(select(_e432_, 0u, _e437_));
                    let _e440_ = (_e234_ + 7u);
                    if _e235_ {
                        phi_752_ = 0u;
                    } else {
                        let _e441_ = (_e116_ - 1u);
                        phi_752_ = select(_e441_, _e440_, (_e440_ < _e441_));
                    }
                    let _e445_ = phi_752_;
                    if (_e445_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e449_ = seed_params.member[_e445_];
                    let _e450_ = bitcast<u32>(_e449_);
                    let _e451_ = (_e450_ & 2139095040u);
                    if (_e451_ == 2139095040u) {
                        phi_770_ = true;
                    } else {
                        phi_770_ = (_e451_ == 0u);
                    }
                    let _e455_ = phi_770_;
                    let _e457_ = bitcast<f32>(select(_e450_, 0u, _e455_));
                    let _e458_ = (_e234_ + 8u);
                    if _e235_ {
                        phi_787_ = 0u;
                    } else {
                        let _e459_ = (_e116_ - 1u);
                        phi_787_ = select(_e459_, _e458_, (_e458_ < _e459_));
                    }
                    let _e463_ = phi_787_;
                    if (_e463_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e467_ = seed_params.member[_e463_];
                    let _e468_ = bitcast<u32>(_e467_);
                    let _e469_ = (_e468_ & 2139095040u);
                    if (_e469_ == 2139095040u) {
                        phi_805_ = true;
                    } else {
                        phi_805_ = (_e469_ == 0u);
                    }
                    let _e473_ = phi_805_;
                    let _e475_ = bitcast<f32>(select(_e468_, 0u, _e473_));
                    let _e476_ = (_e234_ + 9u);
                    if _e235_ {
                        phi_822_ = 0u;
                    } else {
                        let _e477_ = (_e116_ - 1u);
                        phi_822_ = select(_e477_, _e476_, (_e476_ < _e477_));
                    }
                    let _e481_ = phi_822_;
                    if (_e481_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e485_ = seed_params.member[_e481_];
                    let _e486_ = bitcast<u32>(_e485_);
                    let _e487_ = (_e486_ & 2139095040u);
                    if (_e487_ == 2139095040u) {
                        phi_840_ = true;
                    } else {
                        phi_840_ = (_e487_ == 0u);
                    }
                    let _e491_ = phi_840_;
                    let _e493_ = bitcast<f32>(select(_e486_, 0u, _e491_));
                    let _e494_ = (_e234_ + 10u);
                    if _e235_ {
                        phi_857_ = 0u;
                    } else {
                        let _e495_ = (_e116_ - 1u);
                        phi_857_ = select(_e495_, _e494_, (_e494_ < _e495_));
                    }
                    let _e499_ = phi_857_;
                    if (_e499_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e503_ = seed_params.member[_e499_];
                    let _e504_ = bitcast<u32>(_e503_);
                    let _e505_ = (_e504_ & 2139095040u);
                    if (_e505_ == 2139095040u) {
                        phi_875_ = true;
                    } else {
                        phi_875_ = (_e505_ == 0u);
                    }
                    let _e509_ = phi_875_;
                    let _e511_ = bitcast<f32>(select(_e504_, 0u, _e509_));
                    let _e512_ = (_e234_ + 11u);
                    if _e235_ {
                        phi_892_ = 0u;
                    } else {
                        let _e513_ = (_e116_ - 1u);
                        phi_892_ = select(_e513_, _e512_, (_e512_ < _e513_));
                    }
                    let _e517_ = phi_892_;
                    if (_e517_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e521_ = seed_params.member[_e517_];
                    let _e522_ = bitcast<u32>(_e521_);
                    let _e523_ = (_e522_ & 2139095040u);
                    if (_e523_ == 2139095040u) {
                        phi_910_ = true;
                    } else {
                        phi_910_ = (_e523_ == 0u);
                    }
                    let _e527_ = phi_910_;
                    let _e529_ = bitcast<f32>(select(_e522_, 0u, _e527_));
                    let _e535_ = select(select(select(0f, 0.999995f, (0.999995f <= _e439_)), _e439_, (_e439_ <= 0.999995f)), 0f, (_e439_ <= 0f));
                    let _e541_ = select(select(select(0f, 1f, (1f <= _e457_)), _e457_, (_e457_ <= 1f)), 0f, (_e457_ <= 0f));
                    let _e553_ = select(select(select(0f, 64f, (64f <= _e493_)), _e493_, (_e493_ <= 64f)), 0f, (_e493_ <= 0f));
                    let _e559_ = select(select(select(0f, 1f, (1f <= _e511_)), _e511_, (_e511_ <= 1f)), 0f, (_e511_ <= 0f));
                    let _e565_ = select(select(select(0f, 1f, (1f <= _e529_)), _e529_, (_e529_ <= 1f)), 0f, (_e529_ <= 0f));
                    if (_e315_ != _e315_) {
                        phi_7407_ = true;
                    } else {
                        phi_7407_ = (0.00001f >= _e315_);
                    }
                    let _e569_ = phi_7407_;
                    let _e570_ = select(_e315_, 0.00001f, _e569_);
                    if (_e421_ != _e421_) {
                        phi_7467_ = true;
                    } else {
                        phi_7467_ = (0f >= _e421_);
                    }
                    let _e574_ = phi_7467_;
                    let _e575_ = select(_e421_, 0f, _e574_);
                    let _e578_ = (1f + (_e570_ * (_e570_ + _e575_)));
                    let _e581_ = bitcast<f32>((2129859011u - bitcast<u32>(_e578_)));
                    let _e585_ = (_e581_ + (_e581_ * (1f - (_e578_ * _e581_))));
                    let _e589_ = (_e585_ + (_e585_ * (1f - (_e578_ * _e585_))));
                    let _e590_ = (_e570_ * _e589_);
                    let _e591_ = (_e570_ * _e590_);
                    let _e592_ = (_e315_ + _e315_);
                    let _e593_ = bitcast<u32>(_e592_);
                    let _e594_ = (_e593_ & 2139095040u);
                    let _e595_ = (_e594_ == 2139095040u);
                    if _e595_ {
                        if ((_e593_ & 2147483647u) > 2139095040u) {
                            phi_1419_ = f32();
                            phi_1420_ = true;
                        } else {
                            let _e610_ = ((_e593_ & 2147483648u) == 0u);
                            if _e610_ {
                                phi_1417_ = 128f;
                            } else {
                                phi_1417_ = f32();
                            }
                            let _e612_ = phi_1417_;
                            phi_1419_ = _e612_;
                            phi_1420_ = select(true, false, _e610_);
                        }
                        let _e615_ = phi_1419_;
                        let _e617_ = phi_1420_;
                        phi_1499_ = select(_e615_, 0f, _e617_);
                    } else {
                        if (_e592_ < 0f) {
                            phi_1498_ = 0f;
                        } else {
                            if (_e592_ > 128f) {
                                phi_1497_ = 128f;
                            } else {
                                if _e595_ {
                                    phi_1491_ = true;
                                } else {
                                    phi_1491_ = (_e594_ == 0u);
                                }
                                let _e600_ = phi_1491_;
                                phi_1497_ = bitcast<f32>(select(_e593_, 0u, _e600_));
                            }
                            let _e604_ = phi_1497_;
                            phi_1498_ = _e604_;
                        }
                        let _e606_ = phi_1498_;
                        phi_1499_ = _e606_;
                    }
                    let _e620_ = phi_1499_;
                    let _e621_ = (_e234_ + 2u);
                    if _e235_ {
                        phi_1511_ = 0u;
                    } else {
                        let _e622_ = (_e116_ - 1u);
                        phi_1511_ = select(_e622_, _e621_, (_e621_ < _e622_));
                    }
                    let _e626_ = phi_1511_;
                    if (_e626_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e630_ = seed_params.member[_e626_];
                    let _e631_ = bitcast<u32>(_e630_);
                    let _e632_ = (_e631_ & 2139095040u);
                    if (_e632_ == 2139095040u) {
                        phi_1529_ = true;
                    } else {
                        phi_1529_ = (_e632_ == 0u);
                    }
                    let _e636_ = phi_1529_;
                    let _e638_ = bitcast<f32>(select(_e631_, 0u, _e636_));
                    let _e639_ = (_e234_ + 3u);
                    if _e235_ {
                        phi_1546_ = 0u;
                    } else {
                        let _e640_ = (_e116_ - 1u);
                        phi_1546_ = select(_e640_, _e639_, (_e639_ < _e640_));
                    }
                    let _e644_ = phi_1546_;
                    if (_e644_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e648_ = seed_params.member[_e644_];
                    let _e649_ = bitcast<u32>(_e648_);
                    let _e650_ = (_e649_ & 2139095040u);
                    if (_e650_ == 2139095040u) {
                        phi_1564_ = true;
                    } else {
                        phi_1564_ = (_e650_ == 0u);
                    }
                    let _e654_ = phi_1564_;
                    let _e656_ = bitcast<f32>(select(_e649_, 0u, _e654_));
                    let _e657_ = (_e234_ + 4u);
                    if _e235_ {
                        phi_1581_ = 0u;
                    } else {
                        let _e658_ = (_e116_ - 1u);
                        phi_1581_ = select(_e658_, _e657_, (_e657_ < _e658_));
                    }
                    let _e662_ = phi_1581_;
                    if (_e662_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e666_ = seed_params.member[_e662_];
                    let _e667_ = bitcast<u32>(_e666_);
                    let _e668_ = (_e667_ & 2139095040u);
                    if (_e668_ == 2139095040u) {
                        phi_1599_ = true;
                    } else {
                        phi_1599_ = (_e668_ == 0u);
                    }
                    let _e672_ = phi_1599_;
                    let _e674_ = bitcast<f32>(select(_e667_, 0u, _e672_));
                    let _e675_ = (_e234_ + 5u);
                    if _e235_ {
                        phi_1616_ = 0u;
                    } else {
                        let _e676_ = (_e116_ - 1u);
                        phi_1616_ = select(_e676_, _e675_, (_e675_ < _e676_));
                    }
                    let _e680_ = phi_1616_;
                    if (_e680_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e684_ = seed_params.member[_e680_];
                    let _e685_ = bitcast<u32>(_e684_);
                    let _e686_ = (_e685_ & 2139095040u);
                    if (_e686_ == 2139095040u) {
                        phi_1634_ = true;
                    } else {
                        phi_1634_ = (_e686_ == 0u);
                    }
                    let _e690_ = phi_1634_;
                    let _e692_ = bitcast<f32>(select(_e685_, 0u, _e690_));
                    let _e693_ = (_e234_ + 12u);
                    if _e235_ {
                        phi_1651_ = 0u;
                    } else {
                        let _e694_ = (_e116_ - 1u);
                        phi_1651_ = select(_e694_, _e693_, (_e693_ < _e694_));
                    }
                    let _e698_ = phi_1651_;
                    if (_e698_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e702_ = seed_params.member[_e698_];
                    let _e703_ = bitcast<u32>(_e702_);
                    let _e704_ = (_e703_ & 2139095040u);
                    let _e705_ = (_e704_ == 2139095040u);
                    if _e705_ {
                        if ((_e703_ & 2147483647u) > 2139095040u) {
                            phi_1703_ = f32();
                            phi_1704_ = true;
                        } else {
                            let _e720_ = ((_e703_ & 2147483648u) == 0u);
                            if _e720_ {
                                phi_1701_ = 1f;
                            } else {
                                phi_1701_ = f32();
                            }
                            let _e722_ = phi_1701_;
                            phi_1703_ = _e722_;
                            phi_1704_ = select(true, false, _e720_);
                        }
                        let _e725_ = phi_1703_;
                        let _e727_ = phi_1704_;
                        if _e727_ {
                            phi_1724_ = -1f;
                        } else {
                            phi_1724_ = _e725_;
                        }
                        let _e729_ = phi_1724_;
                        phi_1783_ = _e729_;
                    } else {
                        if (_e702_ < -1f) {
                            phi_1782_ = -1f;
                        } else {
                            if (_e702_ > 1f) {
                                phi_1781_ = 1f;
                            } else {
                                if _e705_ {
                                    phi_1775_ = true;
                                } else {
                                    phi_1775_ = (_e704_ == 0u);
                                }
                                let _e710_ = phi_1775_;
                                phi_1781_ = bitcast<f32>(select(_e703_, 0u, _e710_));
                            }
                            let _e714_ = phi_1781_;
                            phi_1782_ = _e714_;
                        }
                        let _e716_ = phi_1782_;
                        phi_1783_ = _e716_;
                    }
                    let _e731_ = phi_1783_;
                    let _e732_ = (_e234_ + 13u);
                    if _e235_ {
                        phi_1795_ = 0u;
                    } else {
                        let _e733_ = (_e116_ - 1u);
                        phi_1795_ = select(_e733_, _e732_, (_e732_ < _e733_));
                    }
                    let _e737_ = phi_1795_;
                    if (_e737_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e741_ = seed_params.member[_e737_];
                    let _e742_ = bitcast<u32>(_e741_);
                    let _e743_ = (_e742_ & 2139095040u);
                    let _e744_ = (_e743_ == 2139095040u);
                    if _e744_ {
                        if ((_e742_ & 2147483647u) > 2139095040u) {
                            phi_1847_ = f32();
                            phi_1848_ = true;
                        } else {
                            let _e759_ = ((_e742_ & 2147483648u) == 0u);
                            if _e759_ {
                                phi_1845_ = 4f;
                            } else {
                                phi_1845_ = f32();
                            }
                            let _e761_ = phi_1845_;
                            phi_1847_ = _e761_;
                            phi_1848_ = select(true, false, _e759_);
                        }
                        let _e764_ = phi_1847_;
                        let _e766_ = phi_1848_;
                        if _e766_ {
                            phi_1868_ = -4f;
                        } else {
                            phi_1868_ = _e764_;
                        }
                        let _e768_ = phi_1868_;
                        phi_1927_ = _e768_;
                    } else {
                        if (_e741_ < -4f) {
                            phi_1926_ = -4f;
                        } else {
                            if (_e741_ > 4f) {
                                phi_1925_ = 4f;
                            } else {
                                if _e744_ {
                                    phi_1919_ = true;
                                } else {
                                    phi_1919_ = (_e743_ == 0u);
                                }
                                let _e749_ = phi_1919_;
                                phi_1925_ = bitcast<f32>(select(_e742_, 0u, _e749_));
                            }
                            let _e753_ = phi_1925_;
                            phi_1926_ = _e753_;
                        }
                        let _e755_ = phi_1926_;
                        phi_1927_ = _e755_;
                    }
                    let _e770_ = phi_1927_;
                    let _e771_ = (_e234_ + 14u);
                    if _e235_ {
                        phi_1939_ = 0u;
                    } else {
                        let _e772_ = (_e116_ - 1u);
                        phi_1939_ = select(_e772_, _e771_, (_e771_ < _e772_));
                    }
                    let _e776_ = phi_1939_;
                    if (_e776_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e780_ = seed_params.member[_e776_];
                    let _e781_ = bitcast<u32>(_e780_);
                    let _e782_ = (_e781_ & 2139095040u);
                    let _e783_ = (_e782_ == 2139095040u);
                    if _e783_ {
                        if ((_e781_ & 2147483647u) > 2139095040u) {
                            phi_1991_ = f32();
                            phi_1992_ = true;
                        } else {
                            let _e798_ = ((_e781_ & 2147483648u) == 0u);
                            if _e798_ {
                                phi_1989_ = 64f;
                            } else {
                                phi_1989_ = f32();
                            }
                            let _e800_ = phi_1989_;
                            phi_1991_ = _e800_;
                            phi_1992_ = select(true, false, _e798_);
                        }
                        let _e803_ = phi_1991_;
                        let _e805_ = phi_1992_;
                        if _e805_ {
                            phi_2012_ = -64f;
                        } else {
                            phi_2012_ = _e803_;
                        }
                        let _e807_ = phi_2012_;
                        phi_2071_ = _e807_;
                    } else {
                        if (_e780_ < -64f) {
                            phi_2070_ = -64f;
                        } else {
                            if (_e780_ > 64f) {
                                phi_2069_ = 64f;
                            } else {
                                if _e783_ {
                                    phi_2063_ = true;
                                } else {
                                    phi_2063_ = (_e782_ == 0u);
                                }
                                let _e788_ = phi_2063_;
                                phi_2069_ = bitcast<f32>(select(_e781_, 0u, _e788_));
                            }
                            let _e792_ = phi_2069_;
                            phi_2070_ = _e792_;
                        }
                        let _e794_ = phi_2070_;
                        phi_2071_ = _e794_;
                    }
                    let _e809_ = phi_2071_;
                    let _e810_ = (_e234_ + 15u);
                    if _e235_ {
                        phi_2083_ = 0u;
                    } else {
                        let _e811_ = (_e116_ - 1u);
                        phi_2083_ = select(_e811_, _e810_, (_e810_ < _e811_));
                    }
                    let _e815_ = phi_2083_;
                    if (_e815_ < _e116_) {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e819_ = seed_params.member[_e815_];
                    let _e820_ = bitcast<u32>(_e819_);
                    let _e821_ = (_e820_ & 2139095040u);
                    let _e822_ = (_e821_ == 2139095040u);
                    if _e822_ {
                        if ((_e820_ & 2147483647u) > 2139095040u) {
                            phi_2135_ = f32();
                            phi_2136_ = true;
                        } else {
                            let _e837_ = ((_e820_ & 2147483648u) == 0u);
                            if _e837_ {
                                phi_2133_ = 64f;
                            } else {
                                phi_2133_ = f32();
                            }
                            let _e839_ = phi_2133_;
                            phi_2135_ = _e839_;
                            phi_2136_ = select(true, false, _e837_);
                        }
                        let _e842_ = phi_2135_;
                        let _e844_ = phi_2136_;
                        if _e844_ {
                            phi_2156_ = -64f;
                        } else {
                            phi_2156_ = _e842_;
                        }
                        let _e846_ = phi_2156_;
                        phi_2215_ = _e846_;
                    } else {
                        if (_e819_ < -64f) {
                            phi_2214_ = -64f;
                        } else {
                            if (_e819_ > 64f) {
                                phi_2213_ = 64f;
                            } else {
                                if _e822_ {
                                    phi_2207_ = true;
                                } else {
                                    phi_2207_ = (_e821_ == 0u);
                                }
                                let _e827_ = phi_2207_;
                                phi_2213_ = bitcast<f32>(select(_e820_, 0u, _e827_));
                            }
                            let _e831_ = phi_2213_;
                            phi_2214_ = _e831_;
                        }
                        let _e833_ = phi_2214_;
                        phi_2215_ = _e833_;
                    }
                    let _e848_ = phi_2215_;
                    if _e155_ {
                        let _e850_ = (_e590_ + _e590_);
                        let _e851_ = (_e591_ + _e591_);
                        phi_2237_ = type_22_(((_e589_ + _e589_) - 1f), -(_e850_), _e850_, (1f - _e851_), (_e850_ * _e191_.x), (_e851_ * _e191_.x));
                    } else {
                        phi_2237_ = type_22_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e860_ = phi_2237_;
                    let _e862_ = select(255u, _e110_, (_e110_ < 255u));
                    let _e864_ = (_e862_ < 1536u);
                    if _e864_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e862_] = _e860_.member;
                    let _e867_ = (256u + _e862_);
                    let _e868_ = (_e867_ < 1536u);
                    if _e868_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e860_.member_1_;
                    let _e871_ = (512u + _e862_);
                    let _e872_ = (_e871_ < 1536u);
                    if _e872_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e871_] = _e860_.member_2_;
                    let _e875_ = (768u + _e862_);
                    let _e876_ = (_e875_ < 1536u);
                    if _e876_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e875_] = _e860_.member_3_;
                    let _e879_ = (1024u + _e862_);
                    let _e880_ = (_e879_ < 1536u);
                    if _e880_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e879_] = _e860_.member_4_;
                    let _e883_ = (1280u + _e862_);
                    let _e884_ = (_e883_ < 1536u);
                    if _e884_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e883_] = _e860_.member_5_;
                    workgroupBarrier();
                    phi_2297_ = 1u;
                    loop {
                        let _e887_ = phi_2297_;
                        let _e888_ = (_e887_ < 256u);
                        if _e888_ {
                            let _e890_ = (_e887_ << bitcast<u32>(1i));
                            let _e893_ = (((_e110_ + 1u) * _e890_) - 1u);
                            if (_e893_ < 256u) {
                                let _e895_ = (_e893_ - _e887_);
                                let _e897_ = select(255u, _e895_, (_e895_ < 255u));
                                if (_e897_ < 1536u) {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e900_ = scan_scratch[_e897_];
                                let _e901_ = (256u + _e897_);
                                if (_e901_ < 1536u) {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e904_ = scan_scratch[_e901_];
                                let _e905_ = (512u + _e897_);
                                if (_e905_ < 1536u) {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e908_ = scan_scratch[_e905_];
                                let _e909_ = (768u + _e897_);
                                if (_e909_ < 1536u) {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e912_ = scan_scratch[_e909_];
                                let _e913_ = (1024u + _e897_);
                                if (_e913_ < 1536u) {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e916_ = scan_scratch[_e913_];
                                let _e917_ = (1280u + _e897_);
                                if (_e917_ < 1536u) {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e920_ = scan_scratch[_e917_];
                                let _e922_ = select(255u, _e893_, (_e893_ < 255u));
                                let _e923_ = (_e922_ < 1536u);
                                if _e923_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e925_ = scan_scratch[_e922_];
                                let _e926_ = (256u + _e922_);
                                let _e927_ = (_e926_ < 1536u);
                                if _e927_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e929_ = scan_scratch[_e926_];
                                let _e930_ = (512u + _e922_);
                                let _e931_ = (_e930_ < 1536u);
                                if _e931_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e933_ = scan_scratch[_e930_];
                                let _e934_ = (768u + _e922_);
                                let _e935_ = (_e934_ < 1536u);
                                if _e935_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e937_ = scan_scratch[_e934_];
                                let _e938_ = (1024u + _e922_);
                                let _e939_ = (_e938_ < 1536u);
                                if _e939_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e941_ = scan_scratch[_e938_];
                                let _e942_ = (1280u + _e922_);
                                let _e943_ = (_e942_ < 1536u);
                                if _e943_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                let _e945_ = scan_scratch[_e942_];
                                if _e923_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                scan_scratch[_e922_] = ((_e925_ * _e900_) + (_e929_ * _e908_));
                                if _e927_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                scan_scratch[_e926_] = ((_e925_ * _e904_) + (_e929_ * _e912_));
                                if _e931_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                scan_scratch[_e930_] = ((_e933_ * _e900_) + (_e937_ * _e908_));
                                if _e935_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                scan_scratch[_e934_] = ((_e933_ * _e904_) + (_e937_ * _e912_));
                                if _e939_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                scan_scratch[_e938_] = (((_e925_ * _e916_) + (_e929_ * _e920_)) + _e941_);
                                if _e943_ {
                                } else {
                                    phi_10187_ = true;
                                    break;
                                }
                                scan_scratch[_e942_] = (((_e933_ * _e916_) + (_e937_ * _e920_)) + _e945_);
                            }
                            workgroupBarrier();
                            phi_2298_ = _e890_;
                        } else {
                            phi_2298_ = u32();
                        }
                        let _e967_ = phi_2298_;
                        phi_2297_ = _e967_;
                        phi_10187_ = _e194_;
                        if !(_e888_) {
                            break;
                        }
                    }
                    let _e970_ = phi_10187_;
                    phi_10743_ = _e970_;
                    if _e970_ {
                        break;
                    }
                    let _e971_ = (_e110_ == 255u);
                    if _e971_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_2560_ = 128u;
                    loop {
                        let _e979_ = phi_2560_;
                        let _e980_ = (_e979_ > 0u);
                        if _e980_ {
                            let _e985_ = (((_e110_ + 1u) * (_e979_ << bitcast<u32>(1i))) - 1u);
                            if (_e985_ < 256u) {
                                let _e987_ = (_e985_ - _e979_);
                                let _e989_ = select(255u, _e987_, (_e987_ < 255u));
                                let _e990_ = (_e989_ < 1536u);
                                if _e990_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e992_ = scan_scratch[_e989_];
                                let _e993_ = (256u + _e989_);
                                let _e994_ = (_e993_ < 1536u);
                                if _e994_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e996_ = scan_scratch[_e993_];
                                let _e997_ = (512u + _e989_);
                                let _e998_ = (_e997_ < 1536u);
                                if _e998_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1000_ = scan_scratch[_e997_];
                                let _e1001_ = (768u + _e989_);
                                let _e1002_ = (_e1001_ < 1536u);
                                if _e1002_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1004_ = scan_scratch[_e1001_];
                                let _e1005_ = (1024u + _e989_);
                                let _e1006_ = (_e1005_ < 1536u);
                                if _e1006_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1008_ = scan_scratch[_e1005_];
                                let _e1009_ = (1280u + _e989_);
                                let _e1010_ = (_e1009_ < 1536u);
                                if _e1010_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1012_ = scan_scratch[_e1009_];
                                let _e1014_ = select(255u, _e985_, (_e985_ < 255u));
                                let _e1015_ = (_e1014_ < 1536u);
                                if _e1015_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1017_ = scan_scratch[_e1014_];
                                let _e1018_ = (256u + _e1014_);
                                let _e1019_ = (_e1018_ < 1536u);
                                if _e1019_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1021_ = scan_scratch[_e1018_];
                                let _e1022_ = (512u + _e1014_);
                                let _e1023_ = (_e1022_ < 1536u);
                                if _e1023_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1025_ = scan_scratch[_e1022_];
                                let _e1026_ = (768u + _e1014_);
                                let _e1027_ = (_e1026_ < 1536u);
                                if _e1027_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1029_ = scan_scratch[_e1026_];
                                let _e1030_ = (1024u + _e1014_);
                                let _e1031_ = (_e1030_ < 1536u);
                                if _e1031_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1033_ = scan_scratch[_e1030_];
                                let _e1034_ = (1280u + _e1014_);
                                let _e1035_ = (_e1034_ < 1536u);
                                if _e1035_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                let _e1037_ = scan_scratch[_e1034_];
                                if _e990_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e989_] = _e1017_;
                                if _e994_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e993_] = _e1021_;
                                if _e998_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e997_] = _e1025_;
                                if _e1002_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e1001_] = _e1029_;
                                if _e1006_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e1005_] = _e1033_;
                                if _e1010_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e1009_] = _e1037_;
                                if _e1015_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e1014_] = ((_e992_ * _e1017_) + (_e996_ * _e1025_));
                                if _e1019_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e1018_] = ((_e992_ * _e1021_) + (_e996_ * _e1029_));
                                if _e1023_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e1022_] = ((_e1000_ * _e1017_) + (_e1004_ * _e1025_));
                                if _e1027_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e1026_] = ((_e1000_ * _e1021_) + (_e1004_ * _e1029_));
                                if _e1031_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e1030_] = (((_e992_ * _e1033_) + (_e996_ * _e1037_)) + _e1008_);
                                if _e1035_ {
                                } else {
                                    phi_10488_ = true;
                                    break;
                                }
                                scan_scratch[_e1034_] = (((_e1000_ * _e1033_) + (_e1004_ * _e1037_)) + _e1012_);
                            }
                            workgroupBarrier();
                            phi_2561_ = (_e979_ >> bitcast<u32>(1i));
                        } else {
                            phi_2561_ = u32();
                        }
                        let _e1061_ = phi_2561_;
                        phi_2560_ = _e1061_;
                        phi_10488_ = _e970_;
                        if !(_e980_) {
                            break;
                        }
                    }
                    let _e1064_ = phi_10488_;
                    phi_10743_ = _e1064_;
                    if _e1064_ {
                        break;
                    }
                    if _e864_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    if _e868_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    if _e872_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    if _e876_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    if _e880_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e1065_ = scan_scratch[_e879_];
                    if _e884_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e1066_ = scan_scratch[_e883_];
                    workgroupBarrier();
                    if _e155_ {
                        let _e1067_ = (_e191_.x - _e1066_);
                        let _e1070_ = ((_e589_ * _e1065_) + (_e590_ * _e1067_));
                        let _e1074_ = ((_e1066_ + (_e590_ * _e1065_)) + (_e591_ * _e1067_));
                        let _e1085_ = bitcast<u32>(((((_e638_ * _e1074_) + (_e656_ * _e1070_)) + (_e674_ * ((_e191_.x - (_e575_ * _e1070_)) - _e1074_))) + (_e692_ * _e191_.x)));
                        let _e1086_ = (_e1085_ & 2139095040u);
                        if (_e1086_ == 2139095040u) {
                            phi_3103_ = true;
                        } else {
                            phi_3103_ = (_e1086_ == 0u);
                        }
                        let _e1090_ = phi_3103_;
                        let _e1093_ = bitcast<u32>(_e1070_);
                        let _e1094_ = (_e1093_ & 2139095040u);
                        if (_e1094_ == 2139095040u) {
                            phi_3119_ = true;
                        } else {
                            phi_3119_ = (_e1094_ == 0u);
                        }
                        let _e1098_ = phi_3119_;
                        phi_3127_ = type_21_(bitcast<f32>(select(_e1085_, 0u, _e1090_)), bitcast<f32>(select(_e1093_, 0u, _e1098_)));
                    } else {
                        phi_3127_ = type_21_(0f, 0f);
                    }
                    let _e1103_ = phi_3127_;
                    if _e155_ {
                        let _e1107_ = (_e590_ + _e590_);
                        let _e1108_ = (_e591_ + _e591_);
                        phi_3150_ = type_22_(((_e589_ + _e589_) - 1f), -(_e1107_), _e1107_, (1f - _e1108_), (_e1107_ * _e191_.y), (_e1108_ * _e191_.y));
                    } else {
                        phi_3150_ = type_22_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e1117_ = phi_3150_;
                    if _e864_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e862_] = _e1117_.member;
                    if _e868_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e1117_.member_1_;
                    if _e872_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e871_] = _e1117_.member_2_;
                    if _e876_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e875_] = _e1117_.member_3_;
                    if _e880_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e879_] = _e1117_.member_4_;
                    if _e884_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e883_] = _e1117_.member_5_;
                    workgroupBarrier();
                    phi_3210_ = 1u;
                    loop {
                        let _e1125_ = phi_3210_;
                        let _e1126_ = (_e1125_ < 256u);
                        if _e1126_ {
                            let _e1128_ = (_e1125_ << bitcast<u32>(1i));
                            let _e1131_ = (((_e110_ + 1u) * _e1128_) - 1u);
                            if (_e1131_ < 256u) {
                                let _e1133_ = (_e1131_ - _e1125_);
                                let _e1135_ = select(255u, _e1133_, (_e1133_ < 255u));
                                if (_e1135_ < 1536u) {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1138_ = scan_scratch[_e1135_];
                                let _e1139_ = (256u + _e1135_);
                                if (_e1139_ < 1536u) {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1142_ = scan_scratch[_e1139_];
                                let _e1143_ = (512u + _e1135_);
                                if (_e1143_ < 1536u) {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1146_ = scan_scratch[_e1143_];
                                let _e1147_ = (768u + _e1135_);
                                if (_e1147_ < 1536u) {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1150_ = scan_scratch[_e1147_];
                                let _e1151_ = (1024u + _e1135_);
                                if (_e1151_ < 1536u) {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1154_ = scan_scratch[_e1151_];
                                let _e1155_ = (1280u + _e1135_);
                                if (_e1155_ < 1536u) {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1158_ = scan_scratch[_e1155_];
                                let _e1160_ = select(255u, _e1131_, (_e1131_ < 255u));
                                let _e1161_ = (_e1160_ < 1536u);
                                if _e1161_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1163_ = scan_scratch[_e1160_];
                                let _e1164_ = (256u + _e1160_);
                                let _e1165_ = (_e1164_ < 1536u);
                                if _e1165_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1167_ = scan_scratch[_e1164_];
                                let _e1168_ = (512u + _e1160_);
                                let _e1169_ = (_e1168_ < 1536u);
                                if _e1169_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1171_ = scan_scratch[_e1168_];
                                let _e1172_ = (768u + _e1160_);
                                let _e1173_ = (_e1172_ < 1536u);
                                if _e1173_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1175_ = scan_scratch[_e1172_];
                                let _e1176_ = (1024u + _e1160_);
                                let _e1177_ = (_e1176_ < 1536u);
                                if _e1177_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1179_ = scan_scratch[_e1176_];
                                let _e1180_ = (1280u + _e1160_);
                                let _e1181_ = (_e1180_ < 1536u);
                                if _e1181_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                let _e1183_ = scan_scratch[_e1180_];
                                if _e1161_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                scan_scratch[_e1160_] = ((_e1163_ * _e1138_) + (_e1167_ * _e1146_));
                                if _e1165_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                scan_scratch[_e1164_] = ((_e1163_ * _e1142_) + (_e1167_ * _e1150_));
                                if _e1169_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                scan_scratch[_e1168_] = ((_e1171_ * _e1138_) + (_e1175_ * _e1146_));
                                if _e1173_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                scan_scratch[_e1172_] = ((_e1171_ * _e1142_) + (_e1175_ * _e1150_));
                                if _e1177_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                scan_scratch[_e1176_] = (((_e1163_ * _e1154_) + (_e1167_ * _e1158_)) + _e1179_);
                                if _e1181_ {
                                } else {
                                    phi_10503_ = true;
                                    break;
                                }
                                scan_scratch[_e1180_] = (((_e1171_ * _e1154_) + (_e1175_ * _e1158_)) + _e1183_);
                            }
                            workgroupBarrier();
                            phi_3211_ = _e1128_;
                        } else {
                            phi_3211_ = u32();
                        }
                        let _e1205_ = phi_3211_;
                        phi_3210_ = _e1205_;
                        phi_10503_ = _e1064_;
                        if !(_e1126_) {
                            break;
                        }
                    }
                    let _e1208_ = phi_10503_;
                    phi_10743_ = _e1208_;
                    if _e1208_ {
                        break;
                    }
                    if _e971_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_3473_ = 128u;
                    loop {
                        let _e1216_ = phi_3473_;
                        let _e1217_ = (_e1216_ > 0u);
                        if _e1217_ {
                            let _e1222_ = (((_e110_ + 1u) * (_e1216_ << bitcast<u32>(1i))) - 1u);
                            if (_e1222_ < 256u) {
                                let _e1224_ = (_e1222_ - _e1216_);
                                let _e1226_ = select(255u, _e1224_, (_e1224_ < 255u));
                                let _e1227_ = (_e1226_ < 1536u);
                                if _e1227_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1229_ = scan_scratch[_e1226_];
                                let _e1230_ = (256u + _e1226_);
                                let _e1231_ = (_e1230_ < 1536u);
                                if _e1231_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1233_ = scan_scratch[_e1230_];
                                let _e1234_ = (512u + _e1226_);
                                let _e1235_ = (_e1234_ < 1536u);
                                if _e1235_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1237_ = scan_scratch[_e1234_];
                                let _e1238_ = (768u + _e1226_);
                                let _e1239_ = (_e1238_ < 1536u);
                                if _e1239_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1241_ = scan_scratch[_e1238_];
                                let _e1242_ = (1024u + _e1226_);
                                let _e1243_ = (_e1242_ < 1536u);
                                if _e1243_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1245_ = scan_scratch[_e1242_];
                                let _e1246_ = (1280u + _e1226_);
                                let _e1247_ = (_e1246_ < 1536u);
                                if _e1247_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1249_ = scan_scratch[_e1246_];
                                let _e1251_ = select(255u, _e1222_, (_e1222_ < 255u));
                                let _e1252_ = (_e1251_ < 1536u);
                                if _e1252_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1254_ = scan_scratch[_e1251_];
                                let _e1255_ = (256u + _e1251_);
                                let _e1256_ = (_e1255_ < 1536u);
                                if _e1256_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1258_ = scan_scratch[_e1255_];
                                let _e1259_ = (512u + _e1251_);
                                let _e1260_ = (_e1259_ < 1536u);
                                if _e1260_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1262_ = scan_scratch[_e1259_];
                                let _e1263_ = (768u + _e1251_);
                                let _e1264_ = (_e1263_ < 1536u);
                                if _e1264_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1266_ = scan_scratch[_e1263_];
                                let _e1267_ = (1024u + _e1251_);
                                let _e1268_ = (_e1267_ < 1536u);
                                if _e1268_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1270_ = scan_scratch[_e1267_];
                                let _e1271_ = (1280u + _e1251_);
                                let _e1272_ = (_e1271_ < 1536u);
                                if _e1272_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                let _e1274_ = scan_scratch[_e1271_];
                                if _e1227_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1226_] = _e1254_;
                                if _e1231_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1230_] = _e1258_;
                                if _e1235_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1234_] = _e1262_;
                                if _e1239_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1238_] = _e1266_;
                                if _e1243_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1242_] = _e1270_;
                                if _e1247_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1246_] = _e1274_;
                                if _e1252_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1251_] = ((_e1229_ * _e1254_) + (_e1233_ * _e1262_));
                                if _e1256_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1255_] = ((_e1229_ * _e1258_) + (_e1233_ * _e1266_));
                                if _e1260_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1259_] = ((_e1237_ * _e1254_) + (_e1241_ * _e1262_));
                                if _e1264_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1263_] = ((_e1237_ * _e1258_) + (_e1241_ * _e1266_));
                                if _e1268_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1267_] = (((_e1229_ * _e1270_) + (_e1233_ * _e1274_)) + _e1245_);
                                if _e1272_ {
                                } else {
                                    phi_10523_ = true;
                                    break;
                                }
                                scan_scratch[_e1271_] = (((_e1237_ * _e1270_) + (_e1241_ * _e1274_)) + _e1249_);
                            }
                            workgroupBarrier();
                            phi_3474_ = (_e1216_ >> bitcast<u32>(1i));
                        } else {
                            phi_3474_ = u32();
                        }
                        let _e1298_ = phi_3474_;
                        phi_3473_ = _e1298_;
                        phi_10523_ = _e1208_;
                        if !(_e1217_) {
                            break;
                        }
                    }
                    let _e1301_ = phi_10523_;
                    phi_10743_ = _e1301_;
                    if _e1301_ {
                        break;
                    }
                    if _e864_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    if _e868_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    if _e872_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    if _e876_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    if _e880_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e1302_ = scan_scratch[_e879_];
                    if _e884_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    let _e1303_ = scan_scratch[_e883_];
                    workgroupBarrier();
                    if _e155_ {
                        let _e1304_ = (_e191_.y - _e1303_);
                        let _e1307_ = ((_e589_ * _e1302_) + (_e590_ * _e1304_));
                        let _e1311_ = ((_e1303_ + (_e590_ * _e1302_)) + (_e591_ * _e1304_));
                        let _e1322_ = bitcast<u32>(((((_e638_ * _e1311_) + (_e656_ * _e1307_)) + (_e674_ * ((_e191_.y - (_e575_ * _e1307_)) - _e1311_))) + (_e692_ * _e191_.y)));
                        let _e1323_ = (_e1322_ & 2139095040u);
                        if (_e1323_ == 2139095040u) {
                            phi_4016_ = true;
                        } else {
                            phi_4016_ = (_e1323_ == 0u);
                        }
                        let _e1327_ = phi_4016_;
                        let _e1330_ = bitcast<u32>(_e1307_);
                        let _e1331_ = (_e1330_ & 2139095040u);
                        if (_e1331_ == 2139095040u) {
                            phi_4032_ = true;
                        } else {
                            phi_4032_ = (_e1331_ == 0u);
                        }
                        let _e1335_ = phi_4032_;
                        phi_4040_ = type_21_(bitcast<f32>(select(_e1322_, 0u, _e1327_)), bitcast<f32>(select(_e1330_, 0u, _e1335_)));
                    } else {
                        phi_4040_ = type_21_(0f, 0f);
                    }
                    let _e1340_ = phi_4040_;
                    if _e864_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e862_] = _e1103_.member;
                    if _e868_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e1340_.member;
                    if _e872_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e871_] = _e1103_.member_1_;
                    if _e876_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e875_] = _e1340_.member_1_;
                    if _e880_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e879_] = 0f;
                    if _e884_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e883_] = 0f;
                    workgroupBarrier();
                    if (_e110_ == 0u) {
                        phi_4105_ = type_15_(0u, select(_e151_, 256u, (256u < _e151_)));
                        phi_4108_ = type_23_(array<f32, 4>(0f, 0f, 0f, 0f));
                        phi_4110_ = type_23_(array<f32, 4>(0f, 0f, 0f, 0f));
                        loop {
                            let _e1346_ = phi_4105_;
                            let _e1348_ = phi_4108_;
                            let _e1350_ = phi_4110_;
                            if (_e1346_.member < _e1346_.member_1_) {
                                phi_4106_ = type_15_((_e1346_.member + 1u), _e1346_.member_1_);
                                phi_4133_ = type_15_(1u, _e1346_.member);
                            } else {
                                phi_4106_ = _e1346_;
                                phi_4133_ = type_15_(0u, u32());
                            }
                            let _e1363_ = phi_4106_;
                            let _e1365_ = phi_4133_;
                            let _e1369_ = (bitcast<i32>(_e1365_.member) != 0i);
                            if _e1369_ {
                                if (_e1365_.member_1_ < 1536u) {
                                } else {
                                    phi_10538_ = true;
                                    break;
                                }
                                let _e1372_ = scan_scratch[_e1365_.member_1_];
                                let _e1373_ = (256u + _e1365_.member_1_);
                                if (_e1373_ < 1536u) {
                                } else {
                                    phi_10538_ = true;
                                    break;
                                }
                                let _e1376_ = scan_scratch[_e1373_];
                                let _e1377_ = (512u + _e1365_.member_1_);
                                if (_e1377_ < 1536u) {
                                } else {
                                    phi_10538_ = true;
                                    break;
                                }
                                let _e1380_ = scan_scratch[_e1377_];
                                let _e1381_ = (768u + _e1365_.member_1_);
                                if (_e1381_ < 1536u) {
                                } else {
                                    phi_10538_ = true;
                                    break;
                                }
                                let _e1384_ = scan_scratch[_e1381_];
                                let _e1385_ = (_e731_ * _e1372_);
                                let _e1386_ = (_e731_ * _e1376_);
                                let _e1389_ = bitcast<u32>((_e620_ * _e1380_));
                                let _e1390_ = (_e1389_ & 2139095040u);
                                let _e1391_ = (_e1390_ == 2139095040u);
                                if _e1391_ {
                                    phi_4186_ = true;
                                } else {
                                    phi_4186_ = (_e1390_ == 0u);
                                }
                                let _e1394_ = phi_4186_;
                                if _e1391_ {
                                    phi_4202_ = true;
                                } else {
                                    phi_4202_ = (_e1390_ == 0u);
                                }
                                let _e1399_ = phi_4202_;
                                if _e1391_ {
                                    phi_4218_ = true;
                                } else {
                                    phi_4218_ = (_e1390_ == 0u);
                                }
                                let _e1404_ = phi_4218_;
                                if _e1391_ {
                                    phi_4234_ = true;
                                } else {
                                    phi_4234_ = (_e1390_ == 0u);
                                }
                                let _e1409_ = phi_4234_;
                                let _e1414_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1389_, 0u, _e1394_))) & 2147483647u));
                                let _e1417_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1389_, 0u, _e1399_))) & 2147483647u));
                                let _e1420_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1389_, 0u, _e1404_))) & 2147483647u));
                                let _e1423_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1389_, 0u, _e1409_))) & 2147483647u));
                                if (_e1414_ != _e1414_) {
                                    phi_7527_ = true;
                                } else {
                                    phi_7527_ = (1000000000000000000f <= _e1414_);
                                }
                                let _e1427_ = phi_7527_;
                                if (_e1417_ != _e1417_) {
                                    phi_7542_ = true;
                                } else {
                                    phi_7542_ = (1000000000000000000f <= _e1417_);
                                }
                                let _e1432_ = phi_7542_;
                                if (_e1420_ != _e1420_) {
                                    phi_7557_ = true;
                                } else {
                                    phi_7557_ = (1000000000000000000f <= _e1420_);
                                }
                                let _e1437_ = phi_7557_;
                                if (_e1423_ != _e1423_) {
                                    phi_7572_ = true;
                                } else {
                                    phi_7572_ = (1000000000000000000f <= _e1423_);
                                }
                                let _e1442_ = phi_7572_;
                                let _e1448_ = (_e553_ <= 0f);
                                let _e1449_ = select((_e553_ * select(_e1414_, 1000000000000000000f, _e1427_)), 0f, _e1448_);
                                let _e1450_ = select((_e553_ * select(_e1417_, 1000000000000000000f, _e1432_)), 0f, _e1448_);
                                let _e1451_ = select((_e553_ * select(_e1420_, 1000000000000000000f, _e1437_)), 0f, _e1448_);
                                let _e1452_ = select((_e553_ * select(_e1423_, 1000000000000000000f, _e1442_)), 0f, _e1448_);
                                let _e1453_ = (1f + _e1449_);
                                let _e1454_ = (1f + _e1450_);
                                let _e1455_ = (1f + _e1451_);
                                let _e1456_ = (1f + _e1452_);
                                let _e1459_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1453_)));
                                let _e1462_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1454_)));
                                let _e1465_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1455_)));
                                let _e1468_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1456_)));
                                let _e1478_ = (_e1459_ + (_e1459_ * (1f - (_e1453_ * _e1459_))));
                                let _e1480_ = (_e1462_ + (_e1462_ * (1f - (_e1454_ * _e1462_))));
                                let _e1482_ = (_e1465_ + (_e1465_ * (1f - (_e1455_ * _e1465_))));
                                let _e1484_ = (_e1468_ + (_e1468_ * (1f - (_e1456_ * _e1468_))));
                                let _e1510_ = (1f - (_e541_ * select((_e1478_ + (_e1478_ * (1f - (_e1453_ * _e1478_)))), 1f, (_e1449_ <= 0f))));
                                let _e1512_ = (1f - (_e541_ * select((_e1480_ + (_e1480_ * (1f - (_e1454_ * _e1480_)))), 1f, (_e1450_ <= 0f))));
                                let _e1514_ = (1f - (_e541_ * select((_e1482_ + (_e1482_ * (1f - (_e1455_ * _e1482_)))), 1f, (_e1451_ <= 0f))));
                                let _e1516_ = (1f - (_e541_ * select((_e1484_ + (_e1484_ * (1f - (_e1456_ * _e1484_)))), 1f, (_e1452_ <= 0f))));
                                if (_e1510_ != _e1510_) {
                                    phi_7587_ = true;
                                } else {
                                    phi_7587_ = (0f >= _e1510_);
                                }
                                let _e1520_ = phi_7587_;
                                let _e1521_ = select(_e1510_, 0f, _e1520_);
                                if (_e1512_ != _e1512_) {
                                    phi_7602_ = true;
                                } else {
                                    phi_7602_ = (0f >= _e1512_);
                                }
                                let _e1525_ = phi_7602_;
                                let _e1526_ = select(_e1512_, 0f, _e1525_);
                                if (_e1514_ != _e1514_) {
                                    phi_7617_ = true;
                                } else {
                                    phi_7617_ = (0f >= _e1514_);
                                }
                                let _e1530_ = phi_7617_;
                                let _e1531_ = select(_e1514_, 0f, _e1530_);
                                if (_e1516_ != _e1516_) {
                                    phi_7632_ = true;
                                } else {
                                    phi_7632_ = (0f >= _e1516_);
                                }
                                let _e1535_ = phi_7632_;
                                let _e1536_ = select(_e1516_, 0f, _e1535_);
                                if (_e1521_ != _e1521_) {
                                    phi_7647_ = true;
                                } else {
                                    phi_7647_ = (1f <= _e1521_);
                                }
                                let _e1540_ = phi_7647_;
                                let _e1541_ = select(_e1521_, 1f, _e1540_);
                                if (_e1526_ != _e1526_) {
                                    phi_7662_ = true;
                                } else {
                                    phi_7662_ = (1f <= _e1526_);
                                }
                                let _e1545_ = phi_7662_;
                                let _e1546_ = select(_e1526_, 1f, _e1545_);
                                if (_e1531_ != _e1531_) {
                                    phi_7677_ = true;
                                } else {
                                    phi_7677_ = (1f <= _e1531_);
                                }
                                let _e1550_ = phi_7677_;
                                let _e1551_ = select(_e1531_, 1f, _e1550_);
                                if (_e1536_ != _e1536_) {
                                    phi_7692_ = true;
                                } else {
                                    phi_7692_ = (1f <= _e1536_);
                                }
                                let _e1555_ = phi_7692_;
                                let _e1556_ = select(_e1536_, 1f, _e1555_);
                                let _e1559_ = (_e535_ * _e1350_.member[0]);
                                let _e1562_ = (_e535_ * _e1350_.member[1]);
                                let _e1565_ = (_e535_ * _e1350_.member[2]);
                                let _e1568_ = (_e535_ * _e1350_.member[3]);
                                let _e1569_ = (1f - _e559_);
                                let _e1570_ = (select(select(select(0f, 1f, (1f <= _e475_)), _e475_, (_e475_ <= 1f)), 0f, (_e475_ <= 0f)) * _e1569_);
                                let _e1572_ = (_e1385_ - (_e1570_ * _e1559_));
                                let _e1574_ = (_e1385_ - (_e1570_ * _e1562_));
                                let _e1576_ = (_e1385_ - (_e1570_ * _e1565_));
                                let _e1578_ = (_e1385_ - (_e1570_ * _e1568_));
                                if (_e1572_ != _e1572_) {
                                    phi_7707_ = true;
                                } else {
                                    phi_7707_ = (-1f >= _e1572_);
                                }
                                let _e1582_ = phi_7707_;
                                let _e1583_ = select(_e1572_, -1f, _e1582_);
                                if (_e1574_ != _e1574_) {
                                    phi_7722_ = true;
                                } else {
                                    phi_7722_ = (-1f >= _e1574_);
                                }
                                let _e1587_ = phi_7722_;
                                let _e1588_ = select(_e1574_, -1f, _e1587_);
                                if (_e1576_ != _e1576_) {
                                    phi_7737_ = true;
                                } else {
                                    phi_7737_ = (-1f >= _e1576_);
                                }
                                let _e1592_ = phi_7737_;
                                let _e1593_ = select(_e1576_, -1f, _e1592_);
                                if (_e1578_ != _e1578_) {
                                    phi_7752_ = true;
                                } else {
                                    phi_7752_ = (-1f >= _e1578_);
                                }
                                let _e1597_ = phi_7752_;
                                let _e1598_ = select(_e1578_, -1f, _e1597_);
                                if (_e1583_ != _e1583_) {
                                    phi_7767_ = true;
                                } else {
                                    phi_7767_ = (1f <= _e1583_);
                                }
                                let _e1602_ = phi_7767_;
                                let _e1603_ = select(_e1583_, 1f, _e1602_);
                                if (_e1588_ != _e1588_) {
                                    phi_7782_ = true;
                                } else {
                                    phi_7782_ = (1f <= _e1588_);
                                }
                                let _e1607_ = phi_7782_;
                                let _e1608_ = select(_e1588_, 1f, _e1607_);
                                if (_e1593_ != _e1593_) {
                                    phi_7797_ = true;
                                } else {
                                    phi_7797_ = (1f <= _e1593_);
                                }
                                let _e1612_ = phi_7797_;
                                let _e1613_ = select(_e1593_, 1f, _e1612_);
                                if (_e1598_ != _e1598_) {
                                    phi_7812_ = true;
                                } else {
                                    phi_7812_ = (1f <= _e1598_);
                                }
                                let _e1617_ = phi_7812_;
                                let _e1618_ = select(_e1598_, 1f, _e1617_);
                                let _e1648_ = select(0f, _e1572_, (bitcast<f32>((bitcast<u32>(_e1572_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e1650_ = select(0f, _e1574_, (bitcast<f32>((bitcast<u32>(_e1574_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e1652_ = select(0f, _e1576_, (bitcast<f32>((bitcast<u32>(_e1576_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e1654_ = select(0f, _e1578_, (bitcast<f32>((bitcast<u32>(_e1578_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e1655_ = (_e1648_ * 499.99997f);
                                let _e1656_ = (_e1650_ * 499.99997f);
                                let _e1657_ = (_e1652_ * 499.99997f);
                                let _e1658_ = (_e1654_ * 499.99997f);
                                if (_e1655_ != _e1655_) {
                                    phi_7827_ = true;
                                } else {
                                    phi_7827_ = (0f >= _e1655_);
                                }
                                let _e1662_ = phi_7827_;
                                let _e1663_ = select(_e1655_, 0f, _e1662_);
                                if (_e1656_ != _e1656_) {
                                    phi_7842_ = true;
                                } else {
                                    phi_7842_ = (0f >= _e1656_);
                                }
                                let _e1667_ = phi_7842_;
                                let _e1668_ = select(_e1656_, 0f, _e1667_);
                                if (_e1657_ != _e1657_) {
                                    phi_7857_ = true;
                                } else {
                                    phi_7857_ = (0f >= _e1657_);
                                }
                                let _e1672_ = phi_7857_;
                                let _e1673_ = select(_e1657_, 0f, _e1672_);
                                if (_e1658_ != _e1658_) {
                                    phi_7872_ = true;
                                } else {
                                    phi_7872_ = (0f >= _e1658_);
                                }
                                let _e1677_ = phi_7872_;
                                let _e1678_ = select(_e1658_, 0f, _e1677_);
                                if (_e1663_ != _e1663_) {
                                    phi_7887_ = true;
                                } else {
                                    phi_7887_ = (1f <= _e1663_);
                                }
                                let _e1682_ = phi_7887_;
                                let _e1683_ = select(_e1663_, 1f, _e1682_);
                                if (_e1668_ != _e1668_) {
                                    phi_7902_ = true;
                                } else {
                                    phi_7902_ = (1f <= _e1668_);
                                }
                                let _e1687_ = phi_7902_;
                                let _e1688_ = select(_e1668_, 1f, _e1687_);
                                if (_e1673_ != _e1673_) {
                                    phi_7917_ = true;
                                } else {
                                    phi_7917_ = (1f <= _e1673_);
                                }
                                let _e1692_ = phi_7917_;
                                let _e1693_ = select(_e1673_, 1f, _e1692_);
                                if (_e1678_ != _e1678_) {
                                    phi_7932_ = true;
                                } else {
                                    phi_7932_ = (1f <= _e1678_);
                                }
                                let _e1697_ = phi_7932_;
                                let _e1698_ = select(_e1678_, 1f, _e1697_);
                                let _e1719_ = (_e1648_ - 0.002f);
                                let _e1720_ = (_e1650_ - 0.002f);
                                let _e1721_ = (_e1652_ - 0.002f);
                                let _e1722_ = (_e1654_ - 0.002f);
                                if (_e1719_ != _e1719_) {
                                    phi_7947_ = true;
                                } else {
                                    phi_7947_ = (0f >= _e1719_);
                                }
                                let _e1726_ = phi_7947_;
                                if (_e1720_ != _e1720_) {
                                    phi_7962_ = true;
                                } else {
                                    phi_7962_ = (0f >= _e1720_);
                                }
                                let _e1731_ = phi_7962_;
                                if (_e1721_ != _e1721_) {
                                    phi_7977_ = true;
                                } else {
                                    phi_7977_ = (0f >= _e1721_);
                                }
                                let _e1736_ = phi_7977_;
                                if (_e1722_ != _e1722_) {
                                    phi_7992_ = true;
                                } else {
                                    phi_7992_ = (0f >= _e1722_);
                                }
                                let _e1741_ = phi_7992_;
                                let _e1743_ = ((0.001f * (((_e1683_ * _e1683_) * _e1683_) * (2f - _e1683_))) + select(_e1719_, 0f, _e1726_));
                                let _e1744_ = ((0.001f * (((_e1688_ * _e1688_) * _e1688_) * (2f - _e1688_))) + select(_e1720_, 0f, _e1731_));
                                let _e1745_ = ((0.001f * (((_e1693_ * _e1693_) * _e1693_) * (2f - _e1693_))) + select(_e1721_, 0f, _e1736_));
                                let _e1746_ = ((0.001f * (((_e1698_ * _e1698_) * _e1698_) * (2f - _e1698_))) + select(_e1722_, 0f, _e1741_));
                                let _e1759_ = (_e559_ <= 0f);
                                let _e1784_ = bitcast<u32>((((1f - (_e559_ * _e1541_)) * _e1559_) + (_e1541_ * select(((_e1569_ * _e1743_) + (_e559_ * (_e1603_ * (1f - (0.33333334f * (_e1603_ * _e1603_)))))), _e1743_, _e1759_))));
                                let _e1785_ = (_e1784_ & 2139095040u);
                                if (_e1785_ == 2139095040u) {
                                    phi_4641_ = true;
                                } else {
                                    phi_4641_ = (_e1785_ == 0u);
                                }
                                let _e1789_ = phi_4641_;
                                let _e1791_ = bitcast<f32>(select(_e1784_, 0u, _e1789_));
                                let _e1792_ = bitcast<u32>((((1f - (_e559_ * _e1546_)) * _e1562_) + (_e1546_ * select(((_e1569_ * _e1744_) + (_e559_ * (_e1608_ * (1f - (0.33333334f * (_e1608_ * _e1608_)))))), _e1744_, _e1759_))));
                                let _e1793_ = (_e1792_ & 2139095040u);
                                if (_e1793_ == 2139095040u) {
                                    phi_4657_ = true;
                                } else {
                                    phi_4657_ = (_e1793_ == 0u);
                                }
                                let _e1797_ = phi_4657_;
                                let _e1800_ = bitcast<u32>((((1f - (_e559_ * _e1551_)) * _e1565_) + (_e1551_ * select(((_e1569_ * _e1745_) + (_e559_ * (_e1613_ * (1f - (0.33333334f * (_e1613_ * _e1613_)))))), _e1745_, _e1759_))));
                                let _e1801_ = (_e1800_ & 2139095040u);
                                if (_e1801_ == 2139095040u) {
                                    phi_4673_ = true;
                                } else {
                                    phi_4673_ = (_e1801_ == 0u);
                                }
                                let _e1805_ = phi_4673_;
                                let _e1808_ = bitcast<u32>((((1f - (_e559_ * _e1556_)) * _e1568_) + (_e1556_ * select(((_e1569_ * _e1746_) + (_e559_ * (_e1618_ * (1f - (0.33333334f * (_e1618_ * _e1618_)))))), _e1746_, _e1759_))));
                                let _e1809_ = (_e1808_ & 2139095040u);
                                if (_e1809_ == 2139095040u) {
                                    phi_4689_ = true;
                                } else {
                                    phi_4689_ = (_e1809_ == 0u);
                                }
                                let _e1813_ = phi_4689_;
                                let _e1818_ = bitcast<u32>((_e620_ * _e1384_));
                                let _e1819_ = (_e1818_ & 2139095040u);
                                let _e1820_ = (_e1819_ == 2139095040u);
                                if _e1820_ {
                                    phi_4711_ = true;
                                } else {
                                    phi_4711_ = (_e1819_ == 0u);
                                }
                                let _e1823_ = phi_4711_;
                                if _e1820_ {
                                    phi_4727_ = true;
                                } else {
                                    phi_4727_ = (_e1819_ == 0u);
                                }
                                let _e1828_ = phi_4727_;
                                if _e1820_ {
                                    phi_4743_ = true;
                                } else {
                                    phi_4743_ = (_e1819_ == 0u);
                                }
                                let _e1833_ = phi_4743_;
                                if _e1820_ {
                                    phi_4759_ = true;
                                } else {
                                    phi_4759_ = (_e1819_ == 0u);
                                }
                                let _e1838_ = phi_4759_;
                                let _e1843_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1818_, 0u, _e1823_))) & 2147483647u));
                                let _e1846_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1818_, 0u, _e1828_))) & 2147483647u));
                                let _e1849_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1818_, 0u, _e1833_))) & 2147483647u));
                                let _e1852_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1818_, 0u, _e1838_))) & 2147483647u));
                                if (_e1843_ != _e1843_) {
                                    phi_8007_ = true;
                                } else {
                                    phi_8007_ = (1000000000000000000f <= _e1843_);
                                }
                                let _e1856_ = phi_8007_;
                                if (_e1846_ != _e1846_) {
                                    phi_8022_ = true;
                                } else {
                                    phi_8022_ = (1000000000000000000f <= _e1846_);
                                }
                                let _e1861_ = phi_8022_;
                                if (_e1849_ != _e1849_) {
                                    phi_8037_ = true;
                                } else {
                                    phi_8037_ = (1000000000000000000f <= _e1849_);
                                }
                                let _e1866_ = phi_8037_;
                                if (_e1852_ != _e1852_) {
                                    phi_8052_ = true;
                                } else {
                                    phi_8052_ = (1000000000000000000f <= _e1852_);
                                }
                                let _e1871_ = phi_8052_;
                                let _e1877_ = select((_e553_ * select(_e1843_, 1000000000000000000f, _e1856_)), 0f, _e1448_);
                                let _e1878_ = select((_e553_ * select(_e1846_, 1000000000000000000f, _e1861_)), 0f, _e1448_);
                                let _e1879_ = select((_e553_ * select(_e1849_, 1000000000000000000f, _e1866_)), 0f, _e1448_);
                                let _e1880_ = select((_e553_ * select(_e1852_, 1000000000000000000f, _e1871_)), 0f, _e1448_);
                                let _e1881_ = (1f + _e1877_);
                                let _e1882_ = (1f + _e1878_);
                                let _e1883_ = (1f + _e1879_);
                                let _e1884_ = (1f + _e1880_);
                                let _e1887_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1881_)));
                                let _e1890_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1882_)));
                                let _e1893_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1883_)));
                                let _e1896_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1884_)));
                                let _e1906_ = (_e1887_ + (_e1887_ * (1f - (_e1881_ * _e1887_))));
                                let _e1908_ = (_e1890_ + (_e1890_ * (1f - (_e1882_ * _e1890_))));
                                let _e1910_ = (_e1893_ + (_e1893_ * (1f - (_e1883_ * _e1893_))));
                                let _e1912_ = (_e1896_ + (_e1896_ * (1f - (_e1884_ * _e1896_))));
                                let _e1938_ = (1f - (_e541_ * select((_e1906_ + (_e1906_ * (1f - (_e1881_ * _e1906_)))), 1f, (_e1877_ <= 0f))));
                                let _e1940_ = (1f - (_e541_ * select((_e1908_ + (_e1908_ * (1f - (_e1882_ * _e1908_)))), 1f, (_e1878_ <= 0f))));
                                let _e1942_ = (1f - (_e541_ * select((_e1910_ + (_e1910_ * (1f - (_e1883_ * _e1910_)))), 1f, (_e1879_ <= 0f))));
                                let _e1944_ = (1f - (_e541_ * select((_e1912_ + (_e1912_ * (1f - (_e1884_ * _e1912_)))), 1f, (_e1880_ <= 0f))));
                                if (_e1938_ != _e1938_) {
                                    phi_8067_ = true;
                                } else {
                                    phi_8067_ = (0f >= _e1938_);
                                }
                                let _e1948_ = phi_8067_;
                                let _e1949_ = select(_e1938_, 0f, _e1948_);
                                if (_e1940_ != _e1940_) {
                                    phi_8082_ = true;
                                } else {
                                    phi_8082_ = (0f >= _e1940_);
                                }
                                let _e1953_ = phi_8082_;
                                let _e1954_ = select(_e1940_, 0f, _e1953_);
                                if (_e1942_ != _e1942_) {
                                    phi_8097_ = true;
                                } else {
                                    phi_8097_ = (0f >= _e1942_);
                                }
                                let _e1958_ = phi_8097_;
                                let _e1959_ = select(_e1942_, 0f, _e1958_);
                                if (_e1944_ != _e1944_) {
                                    phi_8112_ = true;
                                } else {
                                    phi_8112_ = (0f >= _e1944_);
                                }
                                let _e1963_ = phi_8112_;
                                let _e1964_ = select(_e1944_, 0f, _e1963_);
                                if (_e1949_ != _e1949_) {
                                    phi_8127_ = true;
                                } else {
                                    phi_8127_ = (1f <= _e1949_);
                                }
                                let _e1968_ = phi_8127_;
                                let _e1969_ = select(_e1949_, 1f, _e1968_);
                                if (_e1954_ != _e1954_) {
                                    phi_8142_ = true;
                                } else {
                                    phi_8142_ = (1f <= _e1954_);
                                }
                                let _e1973_ = phi_8142_;
                                let _e1974_ = select(_e1954_, 1f, _e1973_);
                                if (_e1959_ != _e1959_) {
                                    phi_8157_ = true;
                                } else {
                                    phi_8157_ = (1f <= _e1959_);
                                }
                                let _e1978_ = phi_8157_;
                                let _e1979_ = select(_e1959_, 1f, _e1978_);
                                if (_e1964_ != _e1964_) {
                                    phi_8172_ = true;
                                } else {
                                    phi_8172_ = (1f <= _e1964_);
                                }
                                let _e1983_ = phi_8172_;
                                let _e1984_ = select(_e1964_, 1f, _e1983_);
                                let _e1987_ = (_e535_ * _e1348_.member[0]);
                                let _e1990_ = (_e535_ * _e1348_.member[1]);
                                let _e1993_ = (_e535_ * _e1348_.member[2]);
                                let _e1996_ = (_e535_ * _e1348_.member[3]);
                                let _e1998_ = (_e1386_ - (_e1570_ * _e1987_));
                                let _e2000_ = (_e1386_ - (_e1570_ * _e1990_));
                                let _e2002_ = (_e1386_ - (_e1570_ * _e1993_));
                                let _e2004_ = (_e1386_ - (_e1570_ * _e1996_));
                                if (_e1998_ != _e1998_) {
                                    phi_8187_ = true;
                                } else {
                                    phi_8187_ = (-1f >= _e1998_);
                                }
                                let _e2008_ = phi_8187_;
                                let _e2009_ = select(_e1998_, -1f, _e2008_);
                                if (_e2000_ != _e2000_) {
                                    phi_8202_ = true;
                                } else {
                                    phi_8202_ = (-1f >= _e2000_);
                                }
                                let _e2013_ = phi_8202_;
                                let _e2014_ = select(_e2000_, -1f, _e2013_);
                                if (_e2002_ != _e2002_) {
                                    phi_8217_ = true;
                                } else {
                                    phi_8217_ = (-1f >= _e2002_);
                                }
                                let _e2018_ = phi_8217_;
                                let _e2019_ = select(_e2002_, -1f, _e2018_);
                                if (_e2004_ != _e2004_) {
                                    phi_8232_ = true;
                                } else {
                                    phi_8232_ = (-1f >= _e2004_);
                                }
                                let _e2023_ = phi_8232_;
                                let _e2024_ = select(_e2004_, -1f, _e2023_);
                                if (_e2009_ != _e2009_) {
                                    phi_8247_ = true;
                                } else {
                                    phi_8247_ = (1f <= _e2009_);
                                }
                                let _e2028_ = phi_8247_;
                                let _e2029_ = select(_e2009_, 1f, _e2028_);
                                if (_e2014_ != _e2014_) {
                                    phi_8262_ = true;
                                } else {
                                    phi_8262_ = (1f <= _e2014_);
                                }
                                let _e2033_ = phi_8262_;
                                let _e2034_ = select(_e2014_, 1f, _e2033_);
                                if (_e2019_ != _e2019_) {
                                    phi_8277_ = true;
                                } else {
                                    phi_8277_ = (1f <= _e2019_);
                                }
                                let _e2038_ = phi_8277_;
                                let _e2039_ = select(_e2019_, 1f, _e2038_);
                                if (_e2024_ != _e2024_) {
                                    phi_8292_ = true;
                                } else {
                                    phi_8292_ = (1f <= _e2024_);
                                }
                                let _e2043_ = phi_8292_;
                                let _e2044_ = select(_e2024_, 1f, _e2043_);
                                let _e2074_ = select(0f, _e1998_, (bitcast<f32>((bitcast<u32>(_e1998_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e2076_ = select(0f, _e2000_, (bitcast<f32>((bitcast<u32>(_e2000_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e2078_ = select(0f, _e2002_, (bitcast<f32>((bitcast<u32>(_e2002_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e2080_ = select(0f, _e2004_, (bitcast<f32>((bitcast<u32>(_e2004_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e2081_ = (_e2074_ * 499.99997f);
                                let _e2082_ = (_e2076_ * 499.99997f);
                                let _e2083_ = (_e2078_ * 499.99997f);
                                let _e2084_ = (_e2080_ * 499.99997f);
                                if (_e2081_ != _e2081_) {
                                    phi_8307_ = true;
                                } else {
                                    phi_8307_ = (0f >= _e2081_);
                                }
                                let _e2088_ = phi_8307_;
                                let _e2089_ = select(_e2081_, 0f, _e2088_);
                                if (_e2082_ != _e2082_) {
                                    phi_8322_ = true;
                                } else {
                                    phi_8322_ = (0f >= _e2082_);
                                }
                                let _e2093_ = phi_8322_;
                                let _e2094_ = select(_e2082_, 0f, _e2093_);
                                if (_e2083_ != _e2083_) {
                                    phi_8337_ = true;
                                } else {
                                    phi_8337_ = (0f >= _e2083_);
                                }
                                let _e2098_ = phi_8337_;
                                let _e2099_ = select(_e2083_, 0f, _e2098_);
                                if (_e2084_ != _e2084_) {
                                    phi_8352_ = true;
                                } else {
                                    phi_8352_ = (0f >= _e2084_);
                                }
                                let _e2103_ = phi_8352_;
                                let _e2104_ = select(_e2084_, 0f, _e2103_);
                                if (_e2089_ != _e2089_) {
                                    phi_8367_ = true;
                                } else {
                                    phi_8367_ = (1f <= _e2089_);
                                }
                                let _e2108_ = phi_8367_;
                                let _e2109_ = select(_e2089_, 1f, _e2108_);
                                if (_e2094_ != _e2094_) {
                                    phi_8382_ = true;
                                } else {
                                    phi_8382_ = (1f <= _e2094_);
                                }
                                let _e2113_ = phi_8382_;
                                let _e2114_ = select(_e2094_, 1f, _e2113_);
                                if (_e2099_ != _e2099_) {
                                    phi_8397_ = true;
                                } else {
                                    phi_8397_ = (1f <= _e2099_);
                                }
                                let _e2118_ = phi_8397_;
                                let _e2119_ = select(_e2099_, 1f, _e2118_);
                                if (_e2104_ != _e2104_) {
                                    phi_8412_ = true;
                                } else {
                                    phi_8412_ = (1f <= _e2104_);
                                }
                                let _e2123_ = phi_8412_;
                                let _e2124_ = select(_e2104_, 1f, _e2123_);
                                let _e2145_ = (_e2074_ - 0.002f);
                                let _e2146_ = (_e2076_ - 0.002f);
                                let _e2147_ = (_e2078_ - 0.002f);
                                let _e2148_ = (_e2080_ - 0.002f);
                                if (_e2145_ != _e2145_) {
                                    phi_8427_ = true;
                                } else {
                                    phi_8427_ = (0f >= _e2145_);
                                }
                                let _e2152_ = phi_8427_;
                                if (_e2146_ != _e2146_) {
                                    phi_8442_ = true;
                                } else {
                                    phi_8442_ = (0f >= _e2146_);
                                }
                                let _e2157_ = phi_8442_;
                                if (_e2147_ != _e2147_) {
                                    phi_8457_ = true;
                                } else {
                                    phi_8457_ = (0f >= _e2147_);
                                }
                                let _e2162_ = phi_8457_;
                                if (_e2148_ != _e2148_) {
                                    phi_8472_ = true;
                                } else {
                                    phi_8472_ = (0f >= _e2148_);
                                }
                                let _e2167_ = phi_8472_;
                                let _e2169_ = ((0.001f * (((_e2109_ * _e2109_) * _e2109_) * (2f - _e2109_))) + select(_e2145_, 0f, _e2152_));
                                let _e2170_ = ((0.001f * (((_e2114_ * _e2114_) * _e2114_) * (2f - _e2114_))) + select(_e2146_, 0f, _e2157_));
                                let _e2171_ = ((0.001f * (((_e2119_ * _e2119_) * _e2119_) * (2f - _e2119_))) + select(_e2147_, 0f, _e2162_));
                                let _e2172_ = ((0.001f * (((_e2124_ * _e2124_) * _e2124_) * (2f - _e2124_))) + select(_e2148_, 0f, _e2167_));
                                let _e2209_ = bitcast<u32>((((1f - (_e559_ * _e1969_)) * _e1987_) + (_e1969_ * select(((_e1569_ * _e2169_) + (_e559_ * (_e2029_ * (1f - (0.33333334f * (_e2029_ * _e2029_)))))), _e2169_, _e1759_))));
                                let _e2210_ = (_e2209_ & 2139095040u);
                                if (_e2210_ == 2139095040u) {
                                    phi_5166_ = true;
                                } else {
                                    phi_5166_ = (_e2210_ == 0u);
                                }
                                let _e2214_ = phi_5166_;
                                let _e2216_ = bitcast<f32>(select(_e2209_, 0u, _e2214_));
                                let _e2217_ = bitcast<u32>((((1f - (_e559_ * _e1974_)) * _e1990_) + (_e1974_ * select(((_e1569_ * _e2170_) + (_e559_ * (_e2034_ * (1f - (0.33333334f * (_e2034_ * _e2034_)))))), _e2170_, _e1759_))));
                                let _e2218_ = (_e2217_ & 2139095040u);
                                if (_e2218_ == 2139095040u) {
                                    phi_5182_ = true;
                                } else {
                                    phi_5182_ = (_e2218_ == 0u);
                                }
                                let _e2222_ = phi_5182_;
                                let _e2225_ = bitcast<u32>((((1f - (_e559_ * _e1979_)) * _e1993_) + (_e1979_ * select(((_e1569_ * _e2171_) + (_e559_ * (_e2039_ * (1f - (0.33333334f * (_e2039_ * _e2039_)))))), _e2171_, _e1759_))));
                                let _e2226_ = (_e2225_ & 2139095040u);
                                if (_e2226_ == 2139095040u) {
                                    phi_5198_ = true;
                                } else {
                                    phi_5198_ = (_e2226_ == 0u);
                                }
                                let _e2230_ = phi_5198_;
                                let _e2233_ = bitcast<u32>((((1f - (_e559_ * _e1984_)) * _e1996_) + (_e1984_ * select(((_e1569_ * _e2172_) + (_e559_ * (_e2044_ * (1f - (0.33333334f * (_e2044_ * _e2044_)))))), _e2172_, _e1759_))));
                                let _e2234_ = (_e2233_ & 2139095040u);
                                if (_e2234_ == 2139095040u) {
                                    phi_5214_ = true;
                                } else {
                                    phi_5214_ = (_e2234_ == 0u);
                                }
                                let _e2238_ = phi_5214_;
                                let _e2244_ = (0.5f * (_e1791_ + _e2216_));
                                let _e2251_ = (1024u + _e1365_.member_1_);
                                if (_e2251_ < 1536u) {
                                } else {
                                    phi_10538_ = true;
                                    break;
                                }
                                scan_scratch[_e2251_] = (_e1791_ + (_e565_ * (_e2244_ - _e1791_)));
                                let _e2254_ = (1280u + _e1365_.member_1_);
                                if (_e2254_ < 1536u) {
                                } else {
                                    phi_10538_ = true;
                                    break;
                                }
                                scan_scratch[_e2254_] = (_e2216_ + (_e565_ * (_e2244_ - _e2216_)));
                                phi_4109_ = type_23_(array<f32, 4>(_e2216_, bitcast<f32>(select(_e2217_, 0u, _e2222_)), bitcast<f32>(select(_e2225_, 0u, _e2230_)), bitcast<f32>(select(_e2233_, 0u, _e2238_))));
                                phi_4111_ = type_23_(array<f32, 4>(_e1791_, bitcast<f32>(select(_e1792_, 0u, _e1797_)), bitcast<f32>(select(_e1800_, 0u, _e1805_)), bitcast<f32>(select(_e1808_, 0u, _e1813_))));
                            } else {
                                phi_4109_ = type_23_();
                                phi_4111_ = type_23_();
                            }
                            let _e2258_ = phi_4109_;
                            let _e2260_ = phi_4111_;
                            phi_4105_ = _e1363_;
                            phi_4108_ = _e2258_;
                            phi_4110_ = _e2260_;
                            phi_10538_ = _e1301_;
                            if !(_e1369_) {
                                break;
                            }
                        }
                        let _e2263_ = phi_10538_;
                        phi_10743_ = _e2263_;
                        if _e2263_ {
                            break;
                        }
                        phi_10966_ = _e2263_;
                    } else {
                        phi_10966_ = _e1301_;
                    }
                    let _e2265_ = phi_10966_;
                    workgroupBarrier();
                    if _e155_ {
                        if _e880_ {
                        } else {
                            phi_10743_ = true;
                            break;
                        }
                        let _e2266_ = scan_scratch[_e879_];
                        phi_5302_ = _e2266_;
                    } else {
                        phi_5302_ = 0f;
                    }
                    let _e2268_ = phi_5302_;
                    if _e155_ {
                        if _e884_ {
                        } else {
                            phi_10743_ = true;
                            break;
                        }
                        let _e2269_ = scan_scratch[_e883_];
                        phi_5314_ = _e2269_;
                    } else {
                        phi_5314_ = 0f;
                    }
                    let _e2271_ = phi_5314_;
                    let _e2273_ = (1f - (_e770_ * _e2268_));
                    if (_e2273_ != _e2273_) {
                        phi_8487_ = true;
                    } else {
                        phi_8487_ = (0.05f >= _e2273_);
                    }
                    let _e2277_ = phi_8487_;
                    let _e2280_ = (1f - (_e770_ * _e2271_));
                    if (_e2280_ != _e2280_) {
                        phi_8502_ = true;
                    } else {
                        phi_8502_ = (0.05f >= _e2280_);
                    }
                    let _e2284_ = phi_8502_;
                    if _e155_ {
                        let _e2289_ = (((_e1103_.member * _e809_) * select(_e2273_, 0.05f, _e2277_)) + (_e848_ * _e2268_));
                        let _e2294_ = select(0f, _e2289_, (bitcast<f32>((bitcast<u32>(_e2289_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                        if (_e2294_ != _e2294_) {
                            phi_8517_ = true;
                        } else {
                            phi_8517_ = (-4096f >= _e2294_);
                        }
                        let _e2298_ = phi_8517_;
                        let _e2299_ = select(_e2294_, -4096f, _e2298_);
                        if (_e2299_ != _e2299_) {
                            phi_8577_ = true;
                        } else {
                            phi_8577_ = (4096f <= _e2299_);
                        }
                        let _e2303_ = phi_8577_;
                        phi_5370_ = select(_e2299_, 4096f, _e2303_);
                    } else {
                        phi_5370_ = 0f;
                    }
                    let _e2306_ = phi_5370_;
                    if _e155_ {
                        let _e2310_ = (((_e1340_.member * _e809_) * select(_e2280_, 0.05f, _e2284_)) + (_e848_ * _e2271_));
                        let _e2315_ = select(0f, _e2310_, (bitcast<f32>((bitcast<u32>(_e2310_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                        if (_e2315_ != _e2315_) {
                            phi_8637_ = true;
                        } else {
                            phi_8637_ = (-4096f >= _e2315_);
                        }
                        let _e2319_ = phi_8637_;
                        let _e2320_ = select(_e2315_, -4096f, _e2319_);
                        if (_e2320_ != _e2320_) {
                            phi_8697_ = true;
                        } else {
                            phi_8697_ = (4096f <= _e2320_);
                        }
                        let _e2324_ = phi_8697_;
                        phi_5420_ = select(_e2320_, 4096f, _e2324_);
                    } else {
                        phi_5420_ = 0f;
                    }
                    let _e2327_ = phi_5420_;
                    workgroupBarrier();
                    if _e864_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e862_] = _e2306_;
                    if _e868_ {
                    } else {
                        phi_10743_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e2327_;
                    workgroupBarrier();
                    let _e2328_ = (_e110_ > 0u);
                    if _e2328_ {
                        if _e155_ {
                            let _e2329_ = (_e110_ - 1u);
                            if (_e2329_ < 1536u) {
                            } else {
                                phi_10743_ = true;
                                break;
                            }
                            let _e2332_ = scan_scratch[_e2329_];
                            phi_5449_ = _e2332_;
                        } else {
                            phi_5449_ = f32();
                        }
                        let _e2334_ = phi_5449_;
                        phi_5451_ = _e2334_;
                        phi_5452_ = select(true, false, _e155_);
                    } else {
                        phi_5451_ = f32();
                        phi_5452_ = true;
                    }
                    let _e2337_ = phi_5451_;
                    let _e2339_ = phi_5452_;
                    let _e2340_ = select(_e2337_, 0f, _e2339_);
                    if _e2328_ {
                        if _e155_ {
                            let _e2341_ = (_e110_ + 255u);
                            if (_e2341_ < 1536u) {
                            } else {
                                phi_10743_ = true;
                                break;
                            }
                            let _e2344_ = scan_scratch[_e2341_];
                            phi_5473_ = _e2344_;
                        } else {
                            phi_5473_ = f32();
                        }
                        let _e2346_ = phi_5473_;
                        phi_5475_ = _e2346_;
                        phi_5476_ = select(true, false, _e155_);
                    } else {
                        phi_5475_ = f32();
                        phi_5476_ = true;
                    }
                    let _e2349_ = phi_5475_;
                    let _e2351_ = phi_5476_;
                    let _e2352_ = select(_e2349_, 0f, _e2351_);
                    let _e2357_ = select(0f, _e2340_, (bitcast<f32>((bitcast<u32>(_e2340_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    if (_e2357_ != _e2357_) {
                        phi_8757_ = true;
                    } else {
                        phi_8757_ = (-4096f >= _e2357_);
                    }
                    let _e2361_ = phi_8757_;
                    let _e2362_ = select(_e2357_, -4096f, _e2361_);
                    if (_e2362_ != _e2362_) {
                        phi_8817_ = true;
                    } else {
                        phi_8817_ = (4096f <= _e2362_);
                    }
                    let _e2366_ = phi_8817_;
                    let _e2367_ = select(_e2362_, 4096f, _e2366_);
                    let _e2372_ = select(0f, _e2306_, (bitcast<f32>((bitcast<u32>(_e2306_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    if (_e2372_ != _e2372_) {
                        phi_8877_ = true;
                    } else {
                        phi_8877_ = (-4096f >= _e2372_);
                    }
                    let _e2376_ = phi_8877_;
                    let _e2377_ = select(_e2372_, -4096f, _e2376_);
                    if (_e2377_ != _e2377_) {
                        phi_8937_ = true;
                    } else {
                        phi_8937_ = (4096f <= _e2377_);
                    }
                    let _e2381_ = phi_8937_;
                    let _e2382_ = select(_e2377_, 4096f, _e2381_);
                    let _e2385_ = (1f + (_e2382_ * _e2382_));
                    let _e2386_ = (1f + (_e2367_ * _e2367_));
                    if (_e2385_ != _e2385_) {
                        phi_8997_ = true;
                    } else {
                        phi_8997_ = (0.000000000001f >= _e2385_);
                    }
                    let _e2390_ = phi_8997_;
                    let _e2391_ = select(_e2385_, 0.000000000001f, _e2390_);
                    let _e2392_ = (_e2391_ * 0.5f);
                    let _e2397_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2391_) >> bitcast<u32>(1i))));
                    let _e2401_ = (_e2397_ * (1.5f - (_e2392_ * (_e2397_ * _e2397_))));
                    let _e2405_ = (_e2401_ * (1.5f - (_e2392_ * (_e2401_ * _e2401_))));
                    let _e2409_ = (_e2405_ * (1.5f - (_e2392_ * (_e2405_ * _e2405_))));
                    if (_e2386_ != _e2386_) {
                        phi_9057_ = true;
                    } else {
                        phi_9057_ = (0.000000000001f >= _e2386_);
                    }
                    let _e2413_ = phi_9057_;
                    let _e2414_ = select(_e2386_, 0.000000000001f, _e2413_);
                    let _e2415_ = (_e2414_ * 0.5f);
                    let _e2420_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2414_) >> bitcast<u32>(1i))));
                    let _e2424_ = (_e2420_ * (1.5f - (_e2415_ * (_e2420_ * _e2420_))));
                    let _e2428_ = (_e2424_ * (1.5f - (_e2415_ * (_e2424_ * _e2424_))));
                    let _e2432_ = (_e2428_ * (1.5f - (_e2415_ * (_e2428_ * _e2428_))));
                    let _e2435_ = ((_e2385_ * _e2409_) + (_e2386_ * _e2432_));
                    let _e2438_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2435_)));
                    let _e2442_ = (_e2438_ + (_e2438_ * (1f - (_e2435_ * _e2438_))));
                    let _e2446_ = (_e2442_ + (_e2442_ * (1f - (_e2435_ * _e2442_))));
                    let _e2456_ = (((_e2382_ + _e2367_) * (_e2446_ + (_e2446_ * (1f - (_e2435_ * _e2446_))))) * (1f + (0.5f * (_e2409_ * _e2432_))));
                    let _e2459_ = bitcast<f32>((bitcast<u32>(_e2367_) & 2147483647u));
                    let _e2462_ = bitcast<f32>((bitcast<u32>(_e2382_) & 2147483647u));
                    if (_e2459_ != _e2459_) {
                        phi_9117_ = true;
                    } else {
                        phi_9117_ = (_e2462_ >= _e2459_);
                    }
                    let _e2466_ = phi_9117_;
                    if (_e2456_ != _e2456_) {
                        phi_9177_ = true;
                    } else {
                        phi_9177_ = (-1f >= _e2456_);
                    }
                    let _e2471_ = phi_9177_;
                    let _e2472_ = select(_e2456_, -1f, _e2471_);
                    if (_e2472_ != _e2472_) {
                        phi_9237_ = true;
                    } else {
                        phi_9237_ = (1f <= _e2472_);
                    }
                    let _e2476_ = phi_9237_;
                    let _e2484_ = select(0f, _e2352_, (bitcast<f32>((bitcast<u32>(_e2352_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    if (_e2484_ != _e2484_) {
                        phi_9417_ = true;
                    } else {
                        phi_9417_ = (-4096f >= _e2484_);
                    }
                    let _e2488_ = phi_9417_;
                    let _e2489_ = select(_e2484_, -4096f, _e2488_);
                    if (_e2489_ != _e2489_) {
                        phi_9477_ = true;
                    } else {
                        phi_9477_ = (4096f <= _e2489_);
                    }
                    let _e2493_ = phi_9477_;
                    let _e2494_ = select(_e2489_, 4096f, _e2493_);
                    let _e2499_ = select(0f, _e2327_, (bitcast<f32>((bitcast<u32>(_e2327_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    if (_e2499_ != _e2499_) {
                        phi_9537_ = true;
                    } else {
                        phi_9537_ = (-4096f >= _e2499_);
                    }
                    let _e2503_ = phi_9537_;
                    let _e2504_ = select(_e2499_, -4096f, _e2503_);
                    if (_e2504_ != _e2504_) {
                        phi_9597_ = true;
                    } else {
                        phi_9597_ = (4096f <= _e2504_);
                    }
                    let _e2508_ = phi_9597_;
                    let _e2509_ = select(_e2504_, 4096f, _e2508_);
                    let _e2512_ = (1f + (_e2509_ * _e2509_));
                    let _e2513_ = (1f + (_e2494_ * _e2494_));
                    if (_e2512_ != _e2512_) {
                        phi_9657_ = true;
                    } else {
                        phi_9657_ = (0.000000000001f >= _e2512_);
                    }
                    let _e2517_ = phi_9657_;
                    let _e2518_ = select(_e2512_, 0.000000000001f, _e2517_);
                    let _e2519_ = (_e2518_ * 0.5f);
                    let _e2524_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2518_) >> bitcast<u32>(1i))));
                    let _e2528_ = (_e2524_ * (1.5f - (_e2519_ * (_e2524_ * _e2524_))));
                    let _e2532_ = (_e2528_ * (1.5f - (_e2519_ * (_e2528_ * _e2528_))));
                    let _e2536_ = (_e2532_ * (1.5f - (_e2519_ * (_e2532_ * _e2532_))));
                    if (_e2513_ != _e2513_) {
                        phi_9717_ = true;
                    } else {
                        phi_9717_ = (0.000000000001f >= _e2513_);
                    }
                    let _e2540_ = phi_9717_;
                    let _e2541_ = select(_e2513_, 0.000000000001f, _e2540_);
                    let _e2542_ = (_e2541_ * 0.5f);
                    let _e2547_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2541_) >> bitcast<u32>(1i))));
                    let _e2551_ = (_e2547_ * (1.5f - (_e2542_ * (_e2547_ * _e2547_))));
                    let _e2555_ = (_e2551_ * (1.5f - (_e2542_ * (_e2551_ * _e2551_))));
                    let _e2559_ = (_e2555_ * (1.5f - (_e2542_ * (_e2555_ * _e2555_))));
                    let _e2562_ = ((_e2512_ * _e2536_) + (_e2513_ * _e2559_));
                    let _e2565_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2562_)));
                    let _e2569_ = (_e2565_ + (_e2565_ * (1f - (_e2562_ * _e2565_))));
                    let _e2573_ = (_e2569_ + (_e2569_ * (1f - (_e2562_ * _e2569_))));
                    let _e2583_ = (((_e2509_ + _e2494_) * (_e2573_ + (_e2573_ * (1f - (_e2562_ * _e2573_))))) * (1f + (0.5f * (_e2536_ * _e2559_))));
                    let _e2586_ = bitcast<f32>((bitcast<u32>(_e2494_) & 2147483647u));
                    let _e2589_ = bitcast<f32>((bitcast<u32>(_e2509_) & 2147483647u));
                    if (_e2586_ != _e2586_) {
                        phi_9777_ = true;
                    } else {
                        phi_9777_ = (_e2589_ >= _e2586_);
                    }
                    let _e2593_ = phi_9777_;
                    if (_e2583_ != _e2583_) {
                        phi_9837_ = true;
                    } else {
                        phi_9837_ = (-1f >= _e2583_);
                    }
                    let _e2598_ = phi_9837_;
                    let _e2599_ = select(_e2583_, -1f, _e2598_);
                    if (_e2599_ != _e2599_) {
                        phi_9897_ = true;
                    } else {
                        phi_9897_ = (1f <= _e2599_);
                    }
                    let _e2603_ = phi_9897_;
                    if _e155_ {
                        phi_6754_ = (_e198_ + select(select(_e2599_, 1f, _e2603_), 0f, (select(_e2586_, _e2589_, _e2593_) <= 0f)));
                        phi_6755_ = (_e200_ + select(select(_e2472_, 1f, _e2476_), 0f, (select(_e2459_, _e2462_, _e2466_) <= 0f)));
                    } else {
                        phi_6754_ = _e198_;
                        phi_6755_ = _e200_;
                    }
                    let _e2610_ = phi_6754_;
                    let _e2612_ = phi_6755_;
                    workgroupBarrier();
                    phi_10744_ = _e2265_;
                    phi_303_ = _e2610_;
                    phi_305_ = _e2612_;
                } else {
                    phi_10744_ = _e194_;
                    phi_303_ = f32();
                    phi_305_ = f32();
                }
                let _e2614_ = phi_10744_;
                let _e2616_ = phi_303_;
                let _e2618_ = phi_305_;
                phi_10383_ = _e2614_;
                phi_299_ = _e213_;
                phi_302_ = _e2616_;
                phi_304_ = _e2618_;
                phi_10743_ = _e2614_;
                if !(_e219_) {
                    break;
                }
            }
            let _e2621_ = phi_10743_;
            if _e2621_ {
                break;
            }
            if _e155_ {
                if (_e114_ == 0u) {
                    phi_6769_ = 0u;
                } else {
                    let _e2623_ = (_e114_ - 1u);
                    phi_6769_ = select(_e2623_, _e157_, (_e157_ < _e2623_));
                }
                let _e2627_ = phi_6769_;
                if (_e2627_ < _e114_) {
                } else {
                    break;
                }
                let _e2632_ = target_audio.member[_e2627_].x;
                let _e2636_ = target_audio.member[_e2627_].y;
                let _e2637_ = bitcast<u32>(_e2632_);
                let _e2638_ = (_e2637_ & 2139095040u);
                if (_e2638_ == 2139095040u) {
                    phi_6792_ = true;
                } else {
                    phi_6792_ = (_e2638_ == 0u);
                }
                let _e2642_ = phi_6792_;
                let _e2645_ = bitcast<u32>(_e2636_);
                let _e2646_ = (_e2645_ & 2139095040u);
                if (_e2646_ == 2139095040u) {
                    phi_6808_ = true;
                } else {
                    phi_6808_ = (_e2646_ == 0u);
                }
                let _e2650_ = phi_6808_;
                phi_6816_ = vec2<f32>(bitcast<f32>(select(_e2637_, 0u, _e2642_)), bitcast<f32>(select(_e2645_, 0u, _e2650_)));
            } else {
                phi_6816_ = vec2<f32>(0f, 0f);
            }
            let _e2655_ = phi_6816_;
            if _e155_ {
                let _e2658_ = local;
                phi_6822_ = (_e2658_ - _e2655_.x);
            } else {
                phi_6822_ = 0f;
            }
            let _e2661_ = phi_6822_;
            if _e155_ {
                let _e2664_ = local_1_;
                phi_6828_ = (_e2664_ - _e2655_.y);
            } else {
                phi_6828_ = 0f;
            }
            let _e2667_ = phi_6828_;
            if _e155_ {
                let _e2671_ = bitcast<u32>(((_e2661_ * _e2661_) + (_e2667_ * _e2667_)));
                let _e2672_ = (_e2671_ & 2139095040u);
                if (_e2672_ == 2139095040u) {
                    phi_6845_ = true;
                } else {
                    phi_6845_ = (_e2672_ == 0u);
                }
                let _e2676_ = phi_6845_;
                phi_6851_ = bitcast<f32>(select(_e2671_, 0u, _e2676_));
            } else {
                phi_6851_ = 0f;
            }
            let _e2680_ = phi_6851_;
            if _e155_ {
                let _e2686_ = bitcast<u32>(((_e2655_.x * _e2655_.x) + (_e2655_.y * _e2655_.y)));
                let _e2687_ = (_e2686_ & 2139095040u);
                if (_e2687_ == 2139095040u) {
                    phi_6872_ = true;
                } else {
                    phi_6872_ = (_e2687_ == 0u);
                }
                let _e2691_ = phi_6872_;
                phi_6878_ = bitcast<f32>(select(_e2686_, 0u, _e2691_));
            } else {
                phi_6878_ = 0f;
            }
            let _e2695_ = phi_6878_;
            let _e2696_ = bitcast<u32>(_e2680_);
            let _e2697_ = (_e2696_ & 2139095040u);
            if (_e2697_ == 2139095040u) {
                phi_6889_ = true;
            } else {
                phi_6889_ = (_e2697_ == 0u);
            }
            let _e2701_ = phi_6889_;
            let _e2704_ = bitcast<u32>(_e2695_);
            let _e2705_ = (_e2704_ & 2139095040u);
            if (_e2705_ == 2139095040u) {
                phi_6905_ = true;
            } else {
                phi_6905_ = (_e2705_ == 0u);
            }
            let _e2709_ = phi_6905_;
            let _e2713_ = select(255u, _e110_, (_e110_ < 255u));
            let _e2714_ = (_e2713_ < 256u);
            if _e2714_ {
            } else {
                break;
            }
            err_scratch[_e2713_] = bitcast<f32>(select(_e2696_, 0u, _e2701_));
            if _e2714_ {
            } else {
                break;
            }
            tgt_scratch[_e2713_] = bitcast<f32>(select(_e2704_, 0u, _e2709_));
            workgroupBarrier();
            phi_6935_ = 128u;
            loop {
                let _e2718_ = phi_6935_;
                let _e2719_ = (_e2718_ > 0u);
                if _e2719_ {
                    if (_e110_ < _e2718_) {
                        let _e2721_ = (_e110_ + _e2718_);
                        let _e2722_ = (_e2721_ < 256u);
                        if _e2722_ {
                        } else {
                            phi_10979_ = true;
                            break;
                        }
                        let _e2724_ = err_scratch[_e2721_];
                        let _e2725_ = (_e110_ < 256u);
                        if _e2725_ {
                        } else {
                            phi_10979_ = true;
                            break;
                        }
                        let _e2727_ = err_scratch[_e110_];
                        err_scratch[_e110_] = (_e2727_ + _e2724_);
                        if _e2722_ {
                        } else {
                            phi_10979_ = true;
                            break;
                        }
                        let _e2730_ = tgt_scratch[_e2721_];
                        if _e2725_ {
                        } else {
                            phi_10979_ = true;
                            break;
                        }
                        let _e2732_ = tgt_scratch[_e110_];
                        tgt_scratch[_e110_] = (_e2732_ + _e2730_);
                    }
                    workgroupBarrier();
                    phi_6936_ = (_e2718_ >> bitcast<u32>(1i));
                } else {
                    phi_6936_ = u32();
                }
                let _e2737_ = phi_6936_;
                phi_6935_ = _e2737_;
                phi_10979_ = _e2621_;
                if !(_e2719_) {
                    break;
                }
            }
            let _e2740_ = phi_10979_;
            if _e2740_ {
                break;
            }
            let _e2742_ = err_scratch[0];
            let _e2743_ = bitcast<u32>(_e2742_);
            let _e2744_ = (_e2743_ & 2139095040u);
            if (_e2744_ == 2139095040u) {
                phi_6996_ = true;
            } else {
                phi_6996_ = (_e2744_ == 0u);
            }
            let _e2748_ = phi_6996_;
            let _e2752_ = tgt_scratch[0];
            let _e2753_ = bitcast<u32>(_e2752_);
            let _e2754_ = (_e2753_ & 2139095040u);
            if (_e2754_ == 2139095040u) {
                phi_7014_ = true;
            } else {
                phi_7014_ = (_e2754_ == 0u);
            }
            let _e2758_ = phi_7014_;
            workgroupBarrier();
            if (_e110_ == 0u) {
                if (_e109_.x < _e126_) {
                    if (_e109_.y < _e146_) {
                        if (_e109_.z < select(1u, _e137_, (1u < _e137_))) {
                            let _e2767_ = uniforms.member.member_5_;
                            if ((bitcast<u32>(_e2767_) & 2139095040u) != 2139095040u) {
                                let _e2771_ = (_e2767_ > 0.000000000001f);
                                phi_7054_ = select(f32(), _e2767_, _e2771_);
                                phi_7055_ = select(true, false, _e2771_);
                            } else {
                                phi_7054_ = f32();
                                phi_7055_ = true;
                            }
                            let _e2775_ = phi_7054_;
                            let _e2777_ = phi_7055_;
                            let _e2779_ = (bitcast<f32>(select(_e2753_, 0u, _e2758_)) + select(_e2775_, 0.000000000001f, _e2777_));
                            if ((bitcast<u32>(_e2779_) & 2139095040u) != 2139095040u) {
                                let _e2783_ = (_e2779_ <= 0f);
                                if _e2783_ {
                                    phi_7153_ = f32();
                                } else {
                                    if (_e2779_ < 0.00001f) {
                                        let _e2815_ = (select(0.000000000000000000000001f, _e2779_, (_e2779_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                        if (_e2815_ != _e2815_) {
                                            phi_10137_ = true;
                                        } else {
                                            phi_10137_ = (0.000001f >= _e2815_);
                                        }
                                        let _e2819_ = phi_10137_;
                                        let _e2820_ = select(_e2815_, 0.000001f, _e2819_);
                                        let _e2823_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2820_)));
                                        let _e2827_ = (_e2823_ + (_e2823_ * (1f - (_e2820_ * _e2823_))));
                                        let _e2831_ = (_e2827_ + (_e2827_ * (1f - (_e2820_ * _e2827_))));
                                        let _e2837_ = bitcast<u32>(((_e2831_ + (_e2831_ * (1f - (_e2820_ * _e2831_)))) * 18446744000000000000f));
                                        let _e2838_ = (_e2837_ & 2139095040u);
                                        if (_e2838_ == 2139095040u) {
                                            phi_7111_ = true;
                                        } else {
                                            phi_7111_ = (_e2838_ == 0u);
                                        }
                                        let _e2842_ = phi_7111_;
                                        phi_7152_ = bitcast<f32>(select(_e2837_, 0u, _e2842_));
                                    } else {
                                        if (_e2779_ != _e2779_) {
                                            phi_10077_ = true;
                                        } else {
                                            phi_10077_ = (0.000001f >= _e2779_);
                                        }
                                        let _e2788_ = phi_10077_;
                                        let _e2789_ = select(_e2779_, 0.000001f, _e2788_);
                                        let _e2792_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2789_)));
                                        let _e2796_ = (_e2792_ + (_e2792_ * (1f - (_e2789_ * _e2792_))));
                                        let _e2800_ = (_e2796_ + (_e2796_ * (1f - (_e2789_ * _e2796_))));
                                        let _e2805_ = bitcast<u32>((_e2800_ + (_e2800_ * (1f - (_e2789_ * _e2800_)))));
                                        let _e2806_ = (_e2805_ & 2139095040u);
                                        if (_e2806_ == 2139095040u) {
                                            phi_7146_ = true;
                                        } else {
                                            phi_7146_ = (_e2806_ == 0u);
                                        }
                                        let _e2810_ = phi_7146_;
                                        phi_7152_ = bitcast<f32>(select(_e2805_, 0u, _e2810_));
                                    }
                                    let _e2846_ = phi_7152_;
                                    phi_7153_ = _e2846_;
                                }
                                let _e2848_ = phi_7153_;
                                phi_7154_ = _e2848_;
                                phi_7155_ = _e2783_;
                            } else {
                                phi_7154_ = f32();
                                phi_7155_ = true;
                            }
                            let _e2850_ = phi_7154_;
                            let _e2852_ = phi_7155_;
                            let _e2855_ = bitcast<u32>((bitcast<f32>(select(_e2743_, 0u, _e2748_)) * select(_e2850_, 0f, _e2852_)));
                            let _e2856_ = (_e2855_ & 2139095040u);
                            if (_e2856_ == 2139095040u) {
                                phi_7171_ = true;
                            } else {
                                phi_7171_ = (_e2856_ == 0u);
                            }
                            let _e2860_ = phi_7171_;
                            let _e2866_ = ((((_e109_.z * _e126_) + _e109_.x) * _e146_) + _e109_.y);
                            if (_e118_ == 0u) {
                                phi_7191_ = 0u;
                            } else {
                                let _e2868_ = (_e118_ - 1u);
                                phi_7191_ = select(_e2868_, _e2866_, (_e2866_ < _e2868_));
                            }
                            let _e2872_ = phi_7191_;
                            if (_e2872_ < _e118_) {
                            } else {
                                break;
                            }
                            seed_losses.member[_e2872_] = bitcast<f32>(select(_e2855_, 0u, _e2860_));
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
